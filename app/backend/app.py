import json
import os
import re
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads backend/.env if present; no-op if the file doesn't
                   # exist, and never overrides real env vars already set
                   # (e.g. on Render), since those take priority by default.
except ImportError:
    pass  # python-dotenv not installed -- fine in production, where env vars
          # are set directly by the platform rather than a .env file anyway.

import joblib
import numpy as np
import torch
torch.set_num_threads(1)  # Render free tier is RAM-constrained (512MB); torch's
                          # default thread pool allocates per-thread work buffers
                          # that add up fast on CPU inference. 1 worker + 1 thread
                          # is also plenty for single-request classify/search calls.
import torch.nn.functional as F
from PIL import Image
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from torchvision import transforms as T
from torchvision.models import resnet18
from torch import nn

BASE_DIR = Path(__file__).resolve().parent

def _first_existing_dir(*candidates):
    """Env var wins if set. Otherwise try each candidate in order and use the
    first that actually exists, so this doesn't have to guess your layout."""
    for c in candidates:
        p = Path(c)
        if p.is_dir():
            return p
    return Path(candidates[0])  # none exist yet -- keep the primary candidate
                                 # so error messages point at a sensible path

# MODELS_DIR / DATA_DIR: previously hardcoded as children of app/, which is
# where your models/data actually live -- that was correct and my last edit
# wrongly "fixed" it to a sibling-folder guess based on the VS Code
# screenshot, which broke classify (ARTICLE_MODEL etc. resolved to None,
# hence 'NoneType' object has no attribute 'exists'). Reverted to check the
# app/ children FIRST, with the sibling folder only as a fallback, so this
# keeps working with your actual layout either way.
MODELS_DIR = Path(os.environ["MODELS_DIR"]) if os.environ.get("MODELS_DIR") else \
    _first_existing_dir(BASE_DIR / "models", BASE_DIR.parent / "models")
DATA_DIR = Path(os.environ["DATA_DIR"]) if os.environ.get("DATA_DIR") else \
    _first_existing_dir(BASE_DIR / "data", BASE_DIR.parent / "data")

# Where the actual catalog image FILES live on disk, e.g.:
#   <data_root>/raw/FashionDataset/train/images_train/<id>.jpg
#   <data_root>/raw/FashionDataset/test/images_test/<id>.jpg
#
# IMPORTANT: this is searched INDEPENDENTLY of DATA_DIR above. DATA_DIR only
# needs to find small processed artifacts (label_encoders.pkl etc.) and may
# resolve to a different folder than the one holding the full raw image
# dataset (e.g. a slimmed-down app/data copied in for deployment, vs. the
# full data/ at the project root). Tying image lookup to DATA_DIR was the
# bug in the previous version -- classify worked (it only needed the small
# folder) while every catalog image still 404'd (the images aren't there).
# All three are also settable directly via env var if none of these guesses
# match your actual layout.
def _images_subdir(root, split, images_folder):
    return Path(root) / "raw" / "FashionDataset" / split / images_folder

TRAIN_IMAGES_DIR = Path(os.environ["TRAIN_IMAGES_DIR"]) if os.environ.get("TRAIN_IMAGES_DIR") else \
    _first_existing_dir(
        _images_subdir(DATA_DIR, "train", "images_train"),
        _images_subdir(BASE_DIR / "data", "train", "images_train"),
        _images_subdir(BASE_DIR.parent / "data", "train", "images_train"),
    )
TEST_IMAGES_DIR = Path(os.environ["TEST_IMAGES_DIR"]) if os.environ.get("TEST_IMAGES_DIR") else \
    _first_existing_dir(
        _images_subdir(DATA_DIR, "test", "images_test"),
        _images_subdir(BASE_DIR / "data", "test", "images_test"),
        _images_subdir(BASE_DIR.parent / "data", "test", "images_test"),
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Exact preprocessing used by the submitted notebooks
# ---------------------------------------------------------------------------

def _find_first(*paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None

CONFIG_PATH = _find_first(
    DATA_DIR / "pipeline_config.json",
    DATA_DIR / "processed" / "pipeline_config.json",
    DATA_DIR / "processed_task4" / "pipeline_config.json",
)

if CONFIG_PATH:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    IMAGE_H = int(config["image"]["height"])
    IMAGE_W = int(config["image"]["width"])
    NORM_MEAN = config["image"]["normalization_mean"]
    NORM_STD = config["image"]["normalization_std"]
else:
    # These are the values expected from the supplied notebooks only as a
    # fallback. Prefer shipping pipeline_config.json beside the backend.
    IMAGE_H, IMAGE_W = 80, 60
    NORM_MEAN = [0.5, 0.5, 0.5]
    NORM_STD = [0.5, 0.5, 0.5]

def to_rgb(img):
    return img.convert("RGB")

EVAL_TRANSFORM = T.Compose([
    to_rgb,
    T.Resize((IMAGE_H, IMAGE_W)),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

# ---------------------------------------------------------------------------
# Model architectures copied from the user's notebooks.
# No pretrained weights are used.
# ---------------------------------------------------------------------------

class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        s = x.mean(dim=(2, 3))
        s = torch.sigmoid(self.fc2(F.silu(self.fc1(s))))
        return x * s[:, :, None, None]


class SEResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, drop=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.se = SqueezeExcite(out_ch)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            if (stride != 1 or in_ch != out_ch) else nn.Identity()
        )

    def forward(self, x):
        out = F.silu(self.bn1(x))
        shortcut = self.shortcut(out if not isinstance(self.shortcut, nn.Identity) else x)
        out = self.conv1(out)
        out = self.conv2(F.silu(self.bn2(out)))
        out = self.se(self.drop(out))
        return out + shortcut


class SEResidualCNN(nn.Module):
    def __init__(self, out_dim=128, widths=(32, 64, 128, 256), drop=0.1):
        super().__init__()
        self.stem = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)
        stages, in_ch = [], widths[0]
        for stage_idx, width in enumerate(widths):
            stride = 1 if stage_idx == 0 else 2
            stages.append(SEResidualBlock(in_ch, width, stride=stride, drop=drop))
            stages.append(SEResidualBlock(width, width, stride=1, drop=drop))
            in_ch = width
        self.stages = nn.Sequential(*stages)
        self.norm = nn.BatchNorm2d(in_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = out_dim
        self.proj = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_ch, out_dim))

    def forward(self, x):
        x = self.stages(self.stem(x))
        x = self.pool(F.silu(self.norm(x))).flatten(1)
        return self.proj(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class ImprovedSmallImageEncoder(nn.Module):
    def __init__(self, out_dim=128, dropout=0.2):
        super().__init__()
        self.out_dim = out_dim
        self.stage1 = ConvBlock(3, 32); self.pool1 = nn.MaxPool2d(2)
        self.stage2 = ConvBlock(32, 64); self.pool2 = nn.MaxPool2d(2)
        self.stage3 = ConvBlock(64, 128); self.pool3 = nn.MaxPool2d(2)
        self.stage4 = ConvBlock(128, 256)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(256, out_dim),
            nn.BatchNorm1d(out_dim), nn.SiLU()
        )

    def forward(self, x):
        x = self.pool1(self.stage1(x))
        x = self.pool2(self.stage2(x))
        x = self.pool3(self.stage3(x))
        return self.proj(self.global_pool(self.stage4(x)).flatten(1))


class SmallImageEncoder(nn.Module):
    def __init__(self, out_dim=128, dropout=0.2):
        super().__init__()
        self.out_dim = out_dim
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),    nn.BatchNorm2d(32),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),   nn.BatchNorm2d(64),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),  nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(256, out_dim),
            nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        return self.proj(self.global_pool(x).flatten(1))


class ResNetStyleCNN(nn.Module):
    """Task 4 ResNet18 trained from scratch (weights=None)."""
    def __init__(self, out_dim=128):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.out_dim = out_dim
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x):
        return self.proj(self.backbone(x))


class ScratchResNet18Encoder(ResNetStyleCNN):
    pass


class ImageOnlyClassifier(nn.Module):
    def __init__(self, encoder, n_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.out_dim, n_classes)

    def forward(self, x):
        return self.head(self.encoder(x))


class Classifier(nn.Module):
    def __init__(self, encoder, n_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.out_dim, n_classes)

    def forward(self, x):
        return self.head(self.encoder(x))


class MultiInputNet(nn.Module):
    def __init__(self, metadata_dim, n_classes, image_encoder):
        super().__init__()
        self.image = image_encoder
        self.meta = nn.Sequential(
            nn.BatchNorm1d(metadata_dim),
            nn.Linear(metadata_dim, 128), nn.ReLU(), nn.Dropout(0.2)
        )
        self.head = nn.Sequential(
            nn.Linear(image_encoder.out_dim + 128, 256),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_classes)
        )

    def forward(self, image, metadata):
        return self.head(torch.cat([self.image(image), self.meta(metadata)], dim=1))


class ImageOnlyNet(nn.Module):
    # Task 3, Section 9.2: gender/usage without metadata. Same 2-layer head
    # shape as MultiInputNet's head, minus the metadata branch -- trained
    # specifically for the no-metadata case, so it's a better fit than
    # feeding MultiInputNet an all-unknown metadata vector.
    def __init__(self, n_classes, image_encoder, dropout=0.3):
        super().__init__()
        self.image = image_encoder
        self.head = nn.Sequential(
            nn.Linear(image_encoder.out_dim, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, n_classes),
        )

    def forward(self, image):
        return self.head(self.image(image))


class EmbeddingNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.out_dim = encoder.out_dim

    def forward(self, x):
        return F.normalize(self.encoder(x), p=2, dim=1)


ENCODERS = {
    "SEResidualCNN": SEResidualCNN,
    "SEResidualEncoder": SEResidualCNN,
    "ImprovedSmallImageEncoder": ImprovedSmallImageEncoder,
    "SmallImageEncoder": SmallImageEncoder,
    "ResNetStyleCNN": ResNetStyleCNN,
    "ScratchResNet18Encoder": ScratchResNet18Encoder,
}

def unwrap_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key], checkpoint
        if all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint, {}
        return {k: v for k, v in checkpoint.items() if torch.is_tensor(v)}, checkpoint
    return checkpoint, {}

def infer_encoder_name(state, prefix):
    keys = [k[len(prefix):] for k in state if k.startswith(prefix)]
    if any(k.startswith("backbone.") for k in keys):
        return "ResNetStyleCNN"
    if any(".se." in k or ".se1." in k for k in keys):
        return "SEResidualCNN"
    if any(k.startswith("stage1.conv1") for k in keys):
        return "ImprovedSmallImageEncoder"
    if any(k.startswith("features.") for k in keys):
        return "SmallImageEncoder"
    raise RuntimeError(f"Cannot infer encoder architecture from state dict under {prefix!r}.")

def infer_out_dim(state, prefix):
    candidates = [
        (k, v) for k, v in state.items()
        if k.startswith(prefix) and k.endswith("weight") and getattr(v, "ndim", 0) == 2
    ]
    if not candidates:
        raise RuntimeError(f"Cannot infer embedding dimension under {prefix!r}.")
    return int(candidates[-1][1].shape[0])

def load_image_classifier(path, encoder_joblib):
    ck = torch.load(path, map_location=DEVICE)
    state, meta = unwrap_state(ck)
    enc_name = meta.get("encoder_class") or meta.get("image_encoder")
    if not isinstance(enc_name, str) or enc_name not in ENCODERS:
        enc_name = infer_encoder_name(state, "encoder.")
    out_dim = int(meta.get("out_dim", infer_out_dim(state, "encoder.")))
    n_classes = int(meta.get("n_classes", len(encoder_joblib.classes_)))
    model = ImageOnlyClassifier(ENCODERS[enc_name](out_dim=out_dim), n_classes)
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()

def load_task3_multiinput(path, encoder_joblib, ohe, metadata_dim):
    ck = torch.load(path, map_location=DEVICE)
    state, meta = unwrap_state(ck)
    enc_name = meta.get("image_encoder")
    if not isinstance(enc_name, str) or enc_name not in ENCODERS:
        # MultiInput state keys use image.<encoder>...
        enc_name = infer_encoder_name(state, "image.")
    out_dim = int(meta.get("out_dim", infer_out_dim(state, "image.")))
    n_classes = int(meta.get("n_classes", len(encoder_joblib.classes_)))
    model = MultiInputNet(metadata_dim, n_classes, ENCODERS[enc_name](out_dim=out_dim))
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()

def load_task3_imageonly(path, encoder_joblib):
    ck = torch.load(path, map_location=DEVICE)
    state, meta = unwrap_state(ck)
    enc_name = meta.get("image_encoder")
    if not isinstance(enc_name, str) or enc_name not in ENCODERS:
        # ImageOnlyNet state keys use image.<encoder>..., same prefix as MultiInputNet.
        enc_name = infer_encoder_name(state, "image.")
    out_dim = int(meta.get("out_dim", infer_out_dim(state, "image.")))
    n_classes = int(meta.get("n_classes", len(encoder_joblib.classes_)))
    model = ImageOnlyNet(n_classes, ENCODERS[enc_name](out_dim=out_dim))
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()

def predict_image_model(model, image_bytes):
    with Image.open(__import__("io").BytesIO(image_bytes)) as im:
        x = EVAL_TRANSFORM(im).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    return probs.detach().cpu().numpy()

# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def required(path, label):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path

# Task 1 / Task 2 / Task 3 artifacts follow the filenames in the notebooks.
ARTICLE_MODEL = _find_first(
    MODELS_DIR / "best_imageonly_articletype.pt",
    MODELS_DIR / "task1_models" / "best_imageonly_articletype.pt",
)
ARTICLE_ENCODER = _find_first(
    MODELS_DIR / "articletype_encoder_task1.joblib",
    MODELS_DIR / "task1_models" / "articletype_encoder_task1.joblib",
)
SEASON_MODEL = _find_first(
    MODELS_DIR / "improved_small_cnn_image_only.pt",
    MODELS_DIR / "task2_models" / "improved_small_cnn_image_only.pt",
)
# Prefer the self-describing Task 3 checkpoints produced by notebook cell 69.
GENDER_MODEL = _find_first(
    MODELS_DIR / "gender_multiinput_smallcnn.pt",
    MODELS_DIR / "task3_models" / "gender_multiinput_smallcnn.pt",
)
USAGE_MODEL = _find_first(
    MODELS_DIR / "usage_multiinput_smallcnn.pt",
    MODELS_DIR / "task3_models" / "usage_multiinput_smallcnn.pt",
)
# Optional: gender/usage WITHOUT metadata. Not required at startup -- if
# missing, classify_bytes() falls back to the multi-input model with an
# all-unknown metadata vector instead of refusing to run.
GENDER_IMAGEONLY_MODEL = _find_first(
    MODELS_DIR / "gender_imageonly_smallcnn.pt",
    MODELS_DIR / "task3_models" / "gender_imageonly_smallcnn.pt",
    MODELS_DIR / "gender_imageonly_best.pt",
    MODELS_DIR / "task3_models" / "image_only" / "gender_imageonly_best.pt",
)
USAGE_IMAGEONLY_MODEL = _find_first(
    MODELS_DIR / "usage_imageonly_smallcnn.pt",
    MODELS_DIR / "task3_models" / "usage_imageonly_smallcnn.pt",
    MODELS_DIR / "usage_imageonly_best.pt",
    MODELS_DIR / "task3_models" / "image_only" / "usage_imageonly_best.pt",
)
GENDER_ENCODER = _find_first(
    MODELS_DIR / "gender_encoder.joblib",
    MODELS_DIR / "task3_models" / "gender_encoder.joblib",
)
USAGE_ENCODER = _find_first(
    MODELS_DIR / "usage_encoder.joblib",
    MODELS_DIR / "task3_models" / "usage_encoder.joblib",
)
OHE_PATH = _find_first(
    MODELS_DIR / "ohe_metadata.joblib",
    MODELS_DIR / "task3_models" / "ohe_metadata.joblib",
)

# Loaders are lazy so the server can still start and return a useful error
# if the user has not copied the trained artifacts yet.
_loaded = False
load_error = None

def load_all():
    global _loaded, load_error
    if _loaded:
        return
    try:
        required(ARTICLE_MODEL, "articleType model")
        required(ARTICLE_ENCODER, "articleType label encoder")
        required(SEASON_MODEL, "season model")
        required(GENDER_MODEL, "gender multi-input model")
        required(USAGE_MODEL, "usage multi-input model")
        required(GENDER_ENCODER, "gender label encoder")
        required(USAGE_ENCODER, "usage label encoder")
        required(OHE_PATH, "Task 3 metadata OneHotEncoder")

        global article_model, season_model, gender_model, usage_model
        global gender_imageonly_model, usage_imageonly_model
        global article_enc, season_enc, gender_enc, usage_enc, ohe

        article_enc = joblib.load(ARTICLE_ENCODER)

        # Season encoder is stored in the shared preprocessing artifact.
        label_encoders_path = _find_first(
            DATA_DIR / "label_encoders.pkl",
            DATA_DIR / "processed" / "label_encoders.pkl",
        )
        required(label_encoders_path, "shared label_encoders.pkl")
        all_enc = joblib.load(label_encoders_path) if label_encoders_path.suffix == ".joblib" else __import__("pickle").load(open(label_encoders_path, "rb"))
        season_enc = all_enc["season"]

        gender_enc = joblib.load(GENDER_ENCODER)
        usage_enc = joblib.load(USAGE_ENCODER)
        ohe = joblib.load(OHE_PATH)

        article_model = load_image_classifier(ARTICLE_MODEL, article_enc)
        season_model = load_image_classifier(SEASON_MODEL, season_enc)

        # The OHE artifact knows its own input columns and output width.
        gender_model = load_task3_multiinput(
            GENDER_MODEL, gender_enc, ohe, len(ohe.get_feature_names_out())
        )
        usage_model = load_task3_multiinput(
            USAGE_MODEL, usage_enc, ohe, len(ohe.get_feature_names_out())
        )

        gender_imageonly_model = None
        usage_imageonly_model = None
        if GENDER_IMAGEONLY_MODEL is not None:
            gender_imageonly_model = load_task3_imageonly(GENDER_IMAGEONLY_MODEL, gender_enc)
            print(f"[classify] gender image-only model loaded from {GENDER_IMAGEONLY_MODEL}")
        else:
            print("[classify] gender image-only checkpoint not found in models/ -- requests "
                  "with incomplete metadata will fall back to the multi-input model with "
                  "unknown-category metadata instead of the dedicated no-metadata model")
        if USAGE_IMAGEONLY_MODEL is not None:
            usage_imageonly_model = load_task3_imageonly(USAGE_IMAGEONLY_MODEL, usage_enc)
            print(f"[classify] usage image-only model loaded from {USAGE_IMAGEONLY_MODEL}")
        else:
            print("[classify] usage image-only checkpoint not found in models/ -- requests "
                  "with incomplete metadata will fall back to the multi-input model with "
                  "unknown-category metadata instead of the dedicated no-metadata model")

        _loaded = True
    except Exception as exc:
        load_error = str(exc)
        raise

# ---------------------------------------------------------------------------
# Classification pipeline
# ---------------------------------------------------------------------------

META_COLS = ["masterCategory", "subCategory", "articleType", "baseColour", "year", "season"]
USER_META_COLS = ["masterCategory", "subCategory", "baseColour", "year"]

def _decode(probs, encoder):
    idx = int(np.argmax(probs))
    return str(encoder.inverse_transform([idx])[0]), float(probs[idx])

def classify_bytes(image_bytes, form):
    load_all()

    # Stage 1: exactly the image-only Task 1/Task 2 models.
    article_probs = predict_image_model(article_model, image_bytes)
    season_probs = predict_image_model(season_model, image_bytes)
    article_label, article_conf = _decode(article_probs, article_enc)
    season_label, season_conf = _decode(season_probs, season_enc)

    # Stage 2: Task 3 was trained on six metadata fields. The frontend supplies
    # four; articleType and season are filled by Stage 1, which is the intended
    # chained inference design described in the Task 3 notebook.
    row = {
        "masterCategory": form.get("masterCategory", "").strip(),
        "subCategory": form.get("subCategory", "").strip(),
        "articleType": article_label,
        "baseColour": form.get("baseColour", "").strip(),
        "year": form.get("year", "").strip(),
        "season": season_label,
    }

    with Image.open(__import__("io").BytesIO(image_bytes)) as im:
        image_tensor = EVAL_TRANSFORM(im).unsqueeze(0).to(DEVICE)

    # Gender/usage use two different models depending on how much metadata
    # the user actually supplied:
    #   - all four user fields present  -> MultiInputNet (image + metadata)
    #   - anything missing              -> ImageOnlyNet (image only), the
    #     model Task 3 trained specifically for the no-metadata case, rather
    #     than feeding MultiInputNet a metadata vector with unknown categories
    all_metadata_present = all(row[k] for k in USER_META_COLS)
    use_multiinput = all_metadata_present
    metadata_mode = "multi_input"

    if use_multiinput:
        # OneHotEncoder(handle_unknown='ignore') is the fitted preprocessing
        # used in Task 3.
        meta_matrix = ohe.transform([[row[c] for c in META_COLS]]).astype(np.float32)
        meta_tensor = torch.from_numpy(meta_matrix).to(DEVICE)
        with torch.no_grad():
            g_probs = torch.softmax(gender_model(image_tensor, meta_tensor), dim=1)[0]
            u_probs = torch.softmax(usage_model(image_tensor, meta_tensor), dim=1)[0]
    elif gender_imageonly_model is not None and usage_imageonly_model is not None:
        metadata_mode = "image_only"
        with torch.no_grad():
            g_probs = torch.softmax(gender_imageonly_model(image_tensor), dim=1)[0]
            u_probs = torch.softmax(usage_imageonly_model(image_tensor), dim=1)[0]
    else:
        # Image-only checkpoints not copied into models/ yet -- fall back to
        # multi-input with whatever partial metadata is available (missing
        # fields become "unknown" categories via handle_unknown='ignore').
        # This is a degraded fallback, not the intended behavior: copy
        # gender_imageonly_smallcnn.pt / usage_imageonly_smallcnn.pt into
        # models/ to get the real no-metadata path.
        metadata_mode = "multi_input_partial_fallback"
        meta_matrix = ohe.transform([[row[c] for c in META_COLS]]).astype(np.float32)
        meta_tensor = torch.from_numpy(meta_matrix).to(DEVICE)
        with torch.no_grad():
            g_probs = torch.softmax(gender_model(image_tensor, meta_tensor), dim=1)[0]
            u_probs = torch.softmax(usage_model(image_tensor, meta_tensor), dim=1)[0]

    gender_label, gender_conf = _decode(g_probs.cpu().numpy(), gender_enc)
    usage_label, usage_conf = _decode(u_probs.cpu().numpy(), usage_enc)

    return {
        "articleType": {"label": article_label, "confidence": round(article_conf, 4)},
        "season": {"label": season_label, "confidence": round(season_conf, 4)},
        "gender": {"label": gender_label, "confidence": round(gender_conf, 4)},
        "usage": {"label": usage_label, "confidence": round(usage_conf, 4)},
        # Kept because the frontend ignores extra keys and the mock contract
        # already documents it.
        "usedMetadata": any(row[k] for k in USER_META_COLS),
        "metadataMode": metadata_mode,  # "multi_input" | "image_only" | "multi_input_partial_fallback"
    }

# ---------------------------------------------------------------------------
# Visual search — Task 4 (current handoff)
# ---------------------------------------------------------------------------
# Task 4 is self-contained in models/task4_model.py.  The selected model is:
#   F: metric learning + ArcFace (ablation rung 5)
# The ArcFace head is training-only; inference uses the 128-D L2-normalised
# encoder stored in task4_visual_search.pt.
#
# IMPORTANT: Task 4 ships its own preprocessing metadata and gallery.  We use
# those artifacts directly instead of re-creating the Task 4 architecture or
# borrowing the shared Task 1-3 preprocessing.

TASK4_DIR = MODELS_DIR
TASK4_MODEL = TASK4_DIR / "task4_visual_search.pt"
TASK4_MODULE = TASK4_DIR / "task4_model.py"
TASK4_ARTIFACTS = TASK4_DIR / "task4_artifacts.json"
EMBEDDINGS_PATH = TASK4_DIR / "task4_gallery_embeddings.npy"
INDEX_PATH = TASK4_DIR / "task4_gallery_index.csv"

search_model = None
search_transform = None
catalog_embeddings = None
catalog_index = None
search_meta = None
task4_module = None
search_error = None

def load_search():
    global search_model, search_transform, catalog_embeddings, catalog_index
    global search_meta, task4_module, search_error

    if search_model is not None:
        return

    try:
        required(TASK4_MODEL, "Task 4 model")
        required(TASK4_MODULE, "Task 4 model module")
        required(EMBEDDINGS_PATH, "Task 4 gallery embeddings")
        required(INDEX_PATH, "Task 4 gallery index")

        # Import the exact inference architecture supplied with Task 4.
        # This avoids duplicating/re-implementing the ArcFace rung architecture
        # in this Flask file.
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "task4_model", TASK4_MODULE
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load Task 4 module: {TASK4_MODULE}")

        task4_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task4_module)

        # task4_model.py loads the exact checkpoint architecture and returns
        # the preprocessing metadata saved with that checkpoint.
        search_model, search_meta = task4_module.load_task4_model(
            TASK4_MODEL, device=str(DEVICE)
        )
        search_transform = task4_module.build_transform(search_meta)

        # The current Task 4 handoff stores both gallery files in models/.
        catalog_embeddings, catalog_index = task4_module.load_gallery(TASK4_DIR)
        catalog_embeddings = np.asarray(catalog_embeddings, dtype=np.float32)

        if catalog_embeddings.ndim != 2:
            raise RuntimeError(
                f"Task 4 gallery embeddings must be 2-D, got {catalog_embeddings.shape}"
            )

        if len(catalog_embeddings) != len(catalog_index):
            raise RuntimeError(
                "Task 4 gallery/index length mismatch: "
                f"{len(catalog_embeddings)} vs {len(catalog_index)}"
            )

        embedding_dim = int(search_model.out_dim)
        if catalog_embeddings.shape[1] != embedding_dim:
            raise RuntimeError(
                "Task 4 embedding dimension mismatch: "
                f"gallery={catalog_embeddings.shape[1]}, model={embedding_dim}"
            )

        if "id" not in catalog_index.columns:
            raise RuntimeError("Task 4 gallery index must contain an 'id' column")

        # Sanity-check the shipped artifact metadata when available.
        if TASK4_ARTIFACTS.exists():
            artifact_meta = json.loads(
                TASK4_ARTIFACTS.read_text(encoding="utf-8")
            )
            selected = artifact_meta.get("selected_approach")
            if selected and "F:" not in str(selected):
                raise RuntimeError(
                    "The Task 4 artifacts do not identify the expected F model: "
                    f"{selected!r}"
                )

    except Exception as exc:
        search_error = str(exc)
        raise

def embed_query(image_bytes):
    load_search()

    from io import BytesIO

    with Image.open(BytesIO(image_bytes)) as im:
        # IMPORTANT: use Task 4's own transform, which comes from the exact
        # checkpoint handoff (80x60 + Task 4 mean/std, no augmentation).
        x = search_transform(im).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = search_model(x).float().cpu().numpy()[0]

    # Task4VisualSearchModel already L2-normalises its output.  Keep this
    # defensive normalisation so the cosine/dot-product assumption is explicit.
    norm = float(np.linalg.norm(emb))
    if norm > 1e-12:
        emb = emb / norm

    return emb.astype(np.float32)

def search_similar_bytes(image_bytes, k=10):
    load_search()

    q = embed_query(image_bytes)
    k = max(1, min(int(k), len(catalog_embeddings)))

    # Use the exact search helper shipped with Task 4: brute-force cosine
    # similarity on L2-normalised embeddings.
    # Returns (1, k) arrays for one query.
    indices, similarities = task4_module.search(
        q[None, :], catalog_embeddings, k=k
    )
    top_indices = indices[0]
    top_scores = similarities[0]

    base_url = os.environ.get("CATALOG_IMAGE_BASE_URL", "").rstrip("/")
    results = []

    for idx, score in zip(top_indices, top_scores):
        row = catalog_index.iloc[int(idx)]
        item_id = str(row["id"])

        if "imageUrl" in catalog_index.columns and str(row.get("imageUrl", "")) not in {"nan", ""}:
            image_url = str(row["imageUrl"])
        elif base_url:
            image_url = f"{base_url}/{item_id}.jpg"
        else:
            # Must be an ABSOLUTE url. A relative "/catalog/..." path resolves
            # against the *frontend's* origin in the browser, not this API's
            # origin -- since config.js points the frontend at a different
            # host/port, that's why images rendered as broken icons even
            # though the JSON (similarity %, articleType) came through fine.
            image_url = f"{request.host_url.rstrip('/')}/catalog/{item_id}.jpg"

        article_type = row.get(
            "articleType_grouped",
            row.get("articleType", "")
        )

        results.append({
            "id": item_id,
            "imageUrl": image_url,
            "similarity": round(float(score), 4),
            "articleType": str(article_type) if article_type is not None else "",
        })

    return results

# ---------------------------------------------------------------------------
# Flask API
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Serving catalog images
# ---------------------------------------------------------------------------
# This is the actual fix for "pictures not loading": search_similar_bytes()
# was returning a URL for a route that didn't exist, so every <img> 404'd.

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

def find_catalog_image(item_id):
    """Look up the image file for a catalog id. Task 4's own embed_images()
    always reads '<id>.jpg' (see task4_model.py), so .jpg is tried first;
    .jpeg/.png are defensive fallbacks only."""
    if not _SAFE_ID_RE.match(item_id):
        return None  # reject anything that isn't a plain id (blocks path traversal)
    for directory in (TRAIN_IMAGES_DIR, TEST_IMAGES_DIR):
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = directory / f"{item_id}{ext}"
            if candidate.exists():
                return candidate
    return None

@app.get("/catalog/<item_id>.jpg")
def catalog_image(item_id):
    path = find_catalog_image(item_id)
    if path is None:
        # A real 404 body (not a bare empty one) so this is diagnosable
        # straight from the browser's Network tab.
        return jsonify(
            error=f"No image file found for catalog id {item_id!r}.",
            checked=[str(TRAIN_IMAGES_DIR), str(TEST_IMAGES_DIR)],
        ), 404
    return send_from_directory(path.parent, path.name, max_age=86400)

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "classification_models_loaded": _loaded,
        "classification_error": load_error,
        "search_models_loaded": search_model is not None,
        "search_error": search_error,
        # Diagnostics for the "images not loading" class of bug -- check this
        # endpoint first. If either *_exists is false, set the matching
        # TRAIN_IMAGES_DIR / TEST_IMAGES_DIR env var to the correct path
        # rather than guessing again.
        "train_images_dir": str(TRAIN_IMAGES_DIR),
        "train_images_dir_exists": TRAIN_IMAGES_DIR.is_dir(),
        "test_images_dir": str(TEST_IMAGES_DIR),
        "test_images_dir_exists": TEST_IMAGES_DIR.is_dir(),
    })

@app.post("/api/classify")
def classify_endpoint():
    file = request.files.get("image")
    if file is None:
        return jsonify(error='No image file was uploaded (expected field "image").'), 400
    try:
        raw = file.read()
        if not raw:
            return jsonify(error="Uploaded image is empty."), 400
        return jsonify(classify_bytes(raw, request.form))
    except Exception as exc:
        return jsonify(error=f"Classification failed: {exc}"), 500

@app.post("/api/search-similar")
def search_endpoint():
    file = request.files.get("image")
    if file is None:
        return jsonify(error='No image file was uploaded (expected field "image").'), 400
    try:
        raw = file.read()
        if not raw:
            return jsonify(error="Uploaded image is empty."), 400
        try:
            k = int(request.form.get("k", os.environ.get("SEARCH_TOP_K", "10")))
        except ValueError:
            k = 10
        return jsonify(results=search_similar_bytes(raw, k))
    except Exception as exc:
        return jsonify(error=f"Similar-image search failed: {exc}"), 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )

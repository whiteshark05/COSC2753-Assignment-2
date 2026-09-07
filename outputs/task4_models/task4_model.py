"""Task 4 — Fashion Visual Search: inference-only model, transform and search helpers.

Task 4 is retrieval, not classification: it fills no column of styles_prediction.csv.
Given a query image it returns a ranked list of visually similar catalogue items, so its
deliverable is (a) an encoder that maps an image to a 128-d unit vector, and (b) a
pre-built gallery index to search against. Both ship alongside this file.

Selected approach: F — metric-learning CNN with batch-hard triplet loss + ArcFace
angular margin (ablation rung 5). Chosen over three alternatives on a four-criterion
rule; see Step 14 of COSC2753_A2_Task4.ipynb. The ArcFace head is a *training-time*
classification head and is deliberately absent here — it plays no part in inference.

Quick start
-----------
    import sys; sys.path.insert(0, "../outputs/task4_models")
    from task4_model import load_task4_model, embed_images, load_gallery, search

    model, meta = load_task4_model("../outputs/task4_models/task4_visual_search.pt")
    gal_emb, gal_index = load_gallery("../outputs/task4_models")

    q = embed_images(model, ["1163", "1164"], "../data/raw/FashionDataset/test/images_test", meta)
    idx, sims = search(q, gal_emb, k=10)
    gal_index.iloc[idx[0]]          # top-10 neighbours of the first query

To slot into Pipeline_simplified.ipynb's registry pattern:
    IMG_ENCODER_REGISTRY["Task4VisualSearchEncoder"] = Task4VisualSearchEncoder
The checkpoint stores that name under the "image_encoder" key, matching Tasks 1-3.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import resnet18

__all__ = ["Task4VisualSearchEncoder", "Task4VisualSearchModel", "load_task4_model",
           "build_transform", "embed_images", "load_gallery", "search"]


class Task4VisualSearchEncoder(nn.Module):
    """resnet18 trained from scratch (weights=None, per the assignment) with a linear
    projection to the embedding dimension. Identical to the encoder used in training."""

    def __init__(self, out_dim=128):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.out_dim = out_dim
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x):
        return self.proj(self.backbone(x))


class Task4VisualSearchModel(nn.Module):
    """L2-normalises the encoder output onto the unit sphere, so cosine similarity is a
    plain dot product and the triplet margin used in training keeps its meaning."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.out_dim = encoder.out_dim

    def forward(self, x):
        return F.normalize(self.encoder(x), p=2, dim=1)


def load_task4_model(checkpoint_path, device="cpu"):
    """Returns (model_in_eval_mode, metadata_dict). The metadata carries the image size
    and normalization statistics the weights were trained with -- use build_transform()
    rather than assuming they match another task's."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "state_dict" not in ckpt:
        raise ValueError(f"{checkpoint_path} is not a Task 4 handoff checkpoint "
                         f"(no 'state_dict' key). Keys: {list(ckpt)[:5]}")
    model = Task4VisualSearchModel(Task4VisualSearchEncoder(ckpt["embedding_dim"]))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device).eval(), ckpt["preprocessing"]


def build_transform(meta):
    """The exact eval-time transform used to produce the shipped gallery embeddings.
    No augmentation: the index must be deterministic."""
    return T.Compose([
        T.Lambda(_to_rgb),
        T.Resize((meta["img_height"], meta["img_width"])),
        T.ToTensor(),
        T.Normalize(mean=meta["normalization_mean"], std=meta["normalization_std"]),
    ])


def _to_rgb(im):
    return im.convert("RGB")


class _ImageIdDataset(Dataset):
    def __init__(self, ids, images_dir, transform):
        self.ids = [str(i) for i in ids]
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        with Image.open(self.images_dir / f"{self.ids[i]}.jpg") as im:
            return self.transform(im), self.ids[i]


@torch.no_grad()
def embed_images(model, ids, images_dir, meta, batch_size=256, device=None):
    """Embed images by id, in the order given. Returns (n, embedding_dim) float32,
    already L2-normalised. Row i corresponds to ids[i] -- shuffle is off throughout."""
    device = device or next(model.parameters()).device
    loader = DataLoader(_ImageIdDataset(ids, images_dir, build_transform(meta)),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    out = [model(imgs.to(device)).float().cpu() for imgs, _ in loader]
    return torch.cat(out).numpy().astype(np.float32)


def load_gallery(models_dir):
    """Returns (embeddings, index_dataframe) for the pre-built catalogue index.
    Row i of the embeddings corresponds to row i of the dataframe."""
    import pandas as pd
    models_dir = Path(models_dir)
    emb = np.load(models_dir / "task4_gallery_embeddings.npy")
    index = pd.read_csv(models_dir / "task4_gallery_index.csv", dtype={"id": str})
    if len(emb) != len(index):
        raise ValueError(f"index/embedding mismatch: {len(emb)} vs {len(index)}")
    return emb, index


def search(query_emb, gallery_emb, k=10, chunk=1024):
    """Exact cosine top-k. Both inputs must be L2-normalised (embed_images already is).
    Returns (indices, similarities), each (n_queries, k), best first."""
    k = min(k, len(gallery_emb))
    idx = np.empty((len(query_emb), k), dtype=np.int64)
    sim = np.empty((len(query_emb), k), dtype=np.float32)
    gt = gallery_emb.T.astype(np.float32)
    for start in range(0, len(query_emb), chunk):
        stop = min(start + chunk, len(query_emb))
        sims = query_emb[start:stop].astype(np.float32) @ gt
        part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1)
        idx[start:stop] = np.take_along_axis(part, order, axis=1)
        sim[start:stop] = np.take_along_axis(sims, idx[start:stop], axis=1)
    return idx, sim

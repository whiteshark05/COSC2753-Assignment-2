"""Build the Task 4 handoff package in outputs/task4_models/, matching the layout
Tasks 1-3 use. The consumer notebook it was originally written for
(COSC2753_A2_Pipeline_simplified.ipynb) has since been removed from the repo; the package
stands on its own through task4_model.py, and Task 4 fills no column of
styles_prediction.csv, so nothing downstream loads these weights automatically.

Produces, from the selected ablation rung's checkpoint:
    task4_visual_search.pt          inference weights + preprocessing, dict-style
    task4_artifacts.json            manifest (architecture, metrics, file map)
    task4_gallery_embeddings.npy    the searchable catalogue index
    task4_gallery_index.csv         id + metadata, row-aligned with the embeddings
    task4_model.py                  (hand-maintained) model, transform, search helpers

Runs on CPU by default so it can't contend with a training run for GPU memory.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"
PROC = REPO / "data" / "processed"      # the team's shared split, same as Tasks 1-3
LOG = REPO / "experiments" / "log"     # where rung_ablation.py writes results
IMAGES = REPO / "data" / "raw" / "FashionDataset" / "train" / "images_train"
OUT = REPO / "outputs" / "task4_models"

SELECTED_TAG = "rung5-arcface-50"
SELECTED_NAME = "F: metric learning + ArcFace (ablation rung 5)"
ALTERNATIVES = {
    "rung0-50": "E: metric learning, batch-hard triplet only (the Step 11 model)",
    "rung6-arc-colour-50": "H: ArcFace + supervised colour block (rung 6)",
    "rung3b-w0.10-50": "G: supervised colour block, weight 0.10 (rung 3b)",
}

sys.path.insert(0, str(OUT))
from task4_model import (Task4VisualSearchEncoder, Task4VisualSearchModel,  # noqa: E402
                         embed_images, load_gallery, load_task4_model, search)


def main(device="cpu"):
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((PROC / "pipeline_config.json").read_text())["image"]
    result = json.loads((LOG / f"result_{SELECTED_TAG}.json").read_text())

    # ── 1. inference checkpoint: encoder weights only, ArcFace head dropped ──
    trained = torch.load(EXP / f"model_{SELECTED_TAG}.pt", map_location="cpu")
    encoder_state = {k: v for k, v in trained.items() if not k.startswith("arc.")}
    dropped = sorted(set(trained) - set(encoder_state))
    emb_dim = int(encoder_state["encoder.proj.weight"].shape[0])

    preprocessing = {
        "img_width": int(cfg["width"]), "img_height": int(cfg["height"]),
        "normalization_mean": [float(v) for v in cfg["normalization_mean"]],
        "normalization_std": [float(v) for v in cfg["normalization_std"]],
    }
    ckpt = {
        "image_encoder": "Task4VisualSearchEncoder",   # key Pipeline's registry looks up
        "state_dict": encoder_state,
        "embedding_dim": emb_dim,
        "preprocessing": preprocessing,
        "similarity": "cosine on L2-normalised embeddings",
        "source_checkpoint": f"experiments/model_{SELECTED_TAG}.pt",
    }
    torch.save(ckpt, OUT / "task4_visual_search.pt")
    print(f"checkpoint: {emb_dim}-d, {len(encoder_state)} tensors, dropped {dropped}")

    # ── 2. the catalogue index: every labelled image, per Step 1's framing ──
    gal = pd.read_csv(PROC / "holdout_metadata" / "articleType_train.csv", dtype={"id": str})
    qry = pd.read_csv(PROC / "holdout_metadata" / "articleType_val.csv", dtype={"id": str})
    catalogue = pd.concat([gal, qry], ignore_index=True)
    # dup_group ships too: the catalogue intentionally contains byte-identical product
    # photos (some with contradictory labels), and the consumer needs to be able to
    # collapse them at display time rather than showing a query its own twin.
    keep = ["id", "dup_group", "articleType_grouped", "baseColour", "gender", "season", "usage"]
    index = catalogue[keep].copy()

    model, meta = load_task4_model(OUT / "task4_visual_search.pt", device=device)
    print(f"embedding {len(index):,} catalogue images on {device}...", flush=True)
    emb = embed_images(model, index["id"].tolist(), IMAGES, meta, device=device)
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), f"embeddings not unit-norm: {norms.min()}-{norms.max()}"

    np.save(OUT / "task4_gallery_embeddings.npy", emb)
    index.to_csv(OUT / "task4_gallery_index.csv", index=False)
    print(f"gallery: {emb.shape} ({emb.nbytes/1e6:.1f} MB)")

    # ── 3. manifest ──
    manifest = {
        "task": "visual_search",
        "task_number": 4,
        "selected_approach": SELECTED_NAME,
        "architecture": "Task4VisualSearchModel(Task4VisualSearchEncoder) — resnet18 from scratch "
                        f"+ Linear(512, {emb_dim}), output L2-normalised",
        "pretrained": False,
        "training_losses": ["batch-hard triplet (margin 0.2, PK sampler P=16 x K=4)",
                            "ArcFace additive angular margin (scale 30, margin 0.2)"],
        "identity_label": "articleType_grouped",
        "embedding_dim": emb_dim,
        "image_size_hw": [preprocessing["img_height"], preprocessing["img_width"]],
        "normalization_mean": preprocessing["normalization_mean"],
        "normalization_std": preprocessing["normalization_std"],
        "similarity": "cosine on L2-normalised embeddings (exact brute-force search)",
        "validation_metrics": {
            "mAP@10": result["mAP10"],
            "mAP@10_CI95": [result["ci_low"], result["ci_high"]],
            "mAP@10_strict": result["mAP10_strict"],
            "P@1": result["P@1"], "P@10": result["P@10"],
            "relevance": "simple = matching articleType_grouped; strict = articleType_grouped AND baseColour",
        },
        "alternatives_considered": {
            name: {"mAP@10": json.loads((LOG / f"result_{tag}.json").read_text())["mAP10"],
                   "mAP@10_strict": json.loads((LOG / f"result_{tag}.json").read_text())["mAP10_strict"]}
            for tag, name in ALTERNATIVES.items() if (LOG / f"result_{tag}.json").exists()
        },
        "evaluation_split": {
            "artifacts": "data/processed",
            "gallery_size": int(len(gal)), "query_size": int(len(qry)),
            "split_fingerprint": json.loads((LOG / "split_fingerprint.json").read_text())["fingerprint"]
                                 if (LOG / "split_fingerprint.json").exists() else None,
            "note": "Task 4 scores against the team's shared split, the same artifacts Tasks 1-3 read. "
                    "Every model was retrained on it rather than re-scored, because a checkpoint "
                    "trained on the earlier data/processed_task4 split had seen rows that are query "
                    "rows here. The metrics still differ in kind from Tasks 1-3 (mAP@10 against "
                    "macro-F1), so compare the split, not the score.",
        },
        "shipped_gallery": {
            "size": int(len(index)),
            "distinct_dup_groups": int(index["dup_group"].nunique()),
            "rows_sharing_a_dup_group": int(len(index) - index["dup_group"].nunique()),
            "duplicate_warning": "Some catalogue entries are byte-identical images under different ids, "
                                 "occasionally with contradictory labels (e.g. ids 1165 and 5062 are the "
                                 "same file, labelled Tshirts/Blue and Watches/Black). They embed to "
                                 "cosine 1.0, so a query can retrieve its own twin at rank 1. Collapse "
                                 "results by dup_group before display; the evaluation split already "
                                 "keeps dup_groups disjoint across gallery and query.",
            "note": "All labelled images (gallery + query). Holding images out is an evaluation "
                    "requirement, not a deployment one: a live catalogue search should return any "
                    "stocked product. No reported metric uses this larger index.",
        },
        "files": {
            "model": "task4_visual_search.pt",
            "module": "task4_model.py",
            "gallery_embeddings": "task4_gallery_embeddings.npy",
            "gallery_index": "task4_gallery_index.csv",
        },
        "fills_prediction_column": False,
        "note": "Retrieval, not classification: Task 4 fills no column of styles_prediction.csv. Its "
                "output is a ranked neighbour list per query. The ArcFace head is training-only and is "
                "not in this checkpoint. Register the encoder with "
                "IMG_ENCODER_REGISTRY['Task4VisualSearchEncoder'] to load it the way Tasks 1-3 load theirs.",
    }
    (OUT / "task4_artifacts.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # ── 4. round-trip: reload from disk exactly as a consumer would ──
    model2, meta2 = load_task4_model(OUT / "task4_visual_search.pt", device=device)
    gal_emb, gal_index = load_gallery(OUT)
    probe_ids = gal_index["id"].head(3).tolist()
    q = embed_images(model2, probe_ids, IMAGES, meta2, device=device)
    idx, sims = search(q, gal_emb, k=5)
    drift = float(np.abs(q - gal_emb[:3]).max())
    print(f"\nround-trip: re-embedding shipped rows reproduces the index (max drift {drift:.2e})")
    for r, qid in enumerate(probe_ids):
        qrow = gal_index.iloc[r]
        hits = ", ".join(f"{gal_index.iloc[i]['articleType_grouped']}/{gal_index.iloc[i]['baseColour']}"
                         f" ({sims[r, c]:.3f})" for c, i in enumerate(idx[r][:3]))
        print(f"  query {qid} ({qrow['articleType_grouped']}/{qrow['baseColour']}) -> {hits}")
    print(f"\nwrote {len(list(OUT.iterdir()))} files to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main(device=sys.argv[1] if len(sys.argv) > 1 else "cpu")

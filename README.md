p# COSC2753 Assignment 2 — Fashion Intelligence System

## 1. Folder structure

Place the dataset so the notebook paths resolve correctly. Put the project files like this:

```
project/
├── FashionDataset/
│   ├── train/
│   │   ├── styles_train.csv
│   │   └── images_train/        (~38,600 .jpg files)
│   └── test/
│       ├── styles_prediction.csv
│       └── images_test/         (~5,800 .jpg files)
├── COSC2753_A2_Preprocessing_Consolidated.ipynb
├── requirements.txt
└── README.md
```

Unzip `A2_Fashion.zip` from Canvas directly into `FashionDataset/` so the subfolders match the names above. Don't rename `styles_prediction.csv` or change its columns — it's the required submission format.

## 2. Environment setup

**Python version:** 3.10 or newer.

**Create a virtual environment** (recommended, keeps this separate from other Python projects):

```bash
python3 -m venv venv

# activate it:
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**Install the required packages:**

```bash
pip install -r requirements.txt
```

If you'd rather install manually instead of using the file:

```bash
pip install pandas numpy matplotlib seaborn pillow scipy scikit-learn torch torchvision jupyter
```

**GPU (optional):** Preprocessing runs fine on CPU. If you have an NVIDIA GPU and want CUDA-accelerated training for the modelling notebooks later, install the CUDA build of PyTorch instead — see https://pytorch.org/get-started/locally/ for the correct command for your system.

## 3. Running the notebook

```bash
jupyter notebook
```

Open `COSC2753_A2_Preprocessing_Consolidated.ipynb` and **run all cells top to bottom** (Kernel → Restart & Run All). The sections must run in order — later sections (e.g. the split) depend on variables created earlier (e.g. `dup_group` from the image duplicate check).

Expect this to take a few minutes: hashing ~38,600 images and computing normalization stats over a 3,000-image sample are the slowest steps.

## 4. What it produces

After running, a `processed/` folder appears alongside the notebook:

```
processed/
├── train_full.csv                      full cleaned train split, all columns
├── val_full.csv                        full cleaned val split, all columns
├── pipeline_config.json                every seed/threshold/decision made, in one file
├── label_encoders.pkl                  fitted LabelEncoder per task
└── holdout_metadata/
    ├── articleType_train.csv / _val.csv
    ├── season_train.csv / _val.csv
    ├── gender_train.csv / _val.csv
    ├── usage_train.csv / _val.csv
    └── label_mappings/
        ├── articleType.json
        ├── season.json
        ├── gender.json
        └── usage.json
```

The four task-modelling notebooks (Task 1–3 classifiers, Task 4 visual search) read from `processed/` rather than the raw CSV — run this preprocessing notebook first, once, before any of them.

## 5. Reproducibility

Everything is seeded with `RANDOM_STATE = 42` (train/val split, image sampling for normalization stats, sampler behaviour). Re-running the notebook from scratch should reproduce the same split and the same files in `processed/`.

## 6. Troubleshooting

| Issue | Likely cause |
|---|---|
| `[MISSING]` printed in Section 2 | `FashionDataset/` isn't in the same folder as the notebook, or a subfolder was renamed — check against the structure in §1. |
| `AssertionError` in the leakage checks (§6.12) | Usually means an earlier cell was skipped or re-run out of order — restart and run all cells top to bottom. |
| Very slow on the image hashing / normalization cells | Expected on first run with the full ~38,600 images — this only happens once per environment. |
| `ModuleNotFoundError` | A package from `requirements.txt` didn't install — re-run `pip install -r requirements.txt` inside the activated virtual environment. |

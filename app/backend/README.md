# Fashion Flask backend

This backend implements the exact frontend contract in `frontend/js/app.js`:

- `POST /api/classify`
- `POST /api/search-similar`
- `GET /health`

It uses the trained PyTorch models and fitted preprocessing artifacts from the supplied notebooks. It does not use a pretrained replacement model.

## Structure

```text
claudev2/
├─ app/
│  └─ frontend/
│     └─ ...
├─ backend/
│  ├─ app.py
│  ├─ requirements.txt
│  ├─ models/
│  │  ├─ best_imageonly_articletype.pt
│  │  ├─ articletype_encoder_task1.joblib
│  │  ├─ improved_small_cnn_image_only.pt
│  │  ├─ gender_multiinput_smallcnn.pt
│  │  ├─ usage_multiinput_smallcnn.pt
│  │  ├─ gender_encoder.joblib
│  │  ├─ usage_encoder.joblib
│  │  ├─ ohe_metadata.joblib
│  │  ├─ task4_visual_search.pt
│  │  └─ task4_visual_search_metadata.json
│  └─ data/
│     ├─ pipeline_config.json
│     ├─ label_encoders.pkl
│     ├─ task4_deployment_embeddings.npy
│     └─ task4_deployment_index.csv
```

## Where to copy the trained artifacts

From the notebooks, the classification files are produced under:

```text
outputs/task1_models/
outputs/task2_models/
outputs/task3_models/
outputs/image_only/
```

For the backend, copy the final files into `backend/models/`.

The shared preprocessing files come from:

```text
data/processed/pipeline_config.json
data/processed/label_encoders.pkl
```

Copy those into `backend/data/`.

Task 4 produces:

```text
models/task4_visual_search.pt
models/task4_visual_search_metadata.json
outputs/task4_deployment_embeddings.npy
outputs/task4_deployment_index.csv
```

Copy them as:

```text
backend/models/task4_visual_search.pt
backend/models/task4_visual_search_metadata.json
backend/data/task4_deployment_embeddings.npy
backend/data/task4_deployment_index.csv
```

## Classification pipeline

The frontend supplies:

```text
masterCategory
subCategory
baseColour
year
```

Task 3's fitted metadata encoder was trained on six columns:

```text
masterCategory
subCategory
articleType
baseColour
year
season
```

Therefore the backend chains the models:

1. image -> trained articleType model
2. image -> trained season model
3. supplied metadata + predicted articleType + predicted season
   -> the trained Task 3 gender model
4. supplied metadata + predicted articleType + predicted season
   -> the trained Task 3 usage model

This is why the frontend only needs to provide the four optional metadata fields.

The response is exactly what `app.js` consumes:

```json
{
  "articleType": {"label": "...", "confidence": 0.91},
  "season": {"label": "...", "confidence": 0.84},
  "gender": {"label": "...", "confidence": 0.88},
  "usage": {"label": "...", "confidence": 0.79},
  "usedMetadata": true
}
```

## Visual search

The backend uses:

- your Task 4 CNN embedding model
- the saved 128-dimensional deployment embeddings
- exact cosine/dot-product ranking
- Top-K, default 10

The response is exactly:

```json
{
  "results": [
    {
      "id": "12345",
      "imageUrl": "...",
      "similarity": 0.9123,
      "articleType": "Tshirts"
    }
  ]
}
```

The notebook's deployment embeddings are already L2-normalised, so cosine similarity is the dot product.

## Catalog image URLs

`task4_deployment_index.csv` contains item IDs and labels, but the notebook does not save a web URL for the images.

Set:

```text
CATALOG_IMAGE_BASE_URL=https://your-image-host.example/catalog
```

Then an item with ID `12345` is returned as:

```text
https://your-image-host.example/catalog/12345.jpg
```

You can instead add an `imageUrl` column to `task4_deployment_index.csv`; the backend will use that column directly.

Do not put the entire FashionDataset into GitHub just to make the search UI work. Host catalog images separately and set `CATALOG_IMAGE_BASE_URL`.

## Local development

From `backend/`:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The API runs at:

```text
http://localhost:5000
```

Test health:

```text
http://localhost:5000/health
```

The frontend's existing `config.js` already points localhost to port 5000.

## Render

Create a Render Web Service using the backend directory as the root directory.

Build command:

```text
pip install -r requirements.txt
```

Start command:

```text
gunicorn app:app
```

The code reads:

```python
os.environ["PORT"]
```

and falls back to `5000` locally.

Set the Render environment variable:

```text
CATALOG_IMAGE_BASE_URL=https://your-image-host.example/catalog
```

## Important Task 4 checkpoint note

The supplied Task 4 notebook can promote ablation rungs F/G/H. The backend intentionally refuses to pretend those checkpoints are the base E model, because doing so would violate the requirement to use the existing trained model.

If `task4_visual_search_metadata.json` says:

```text
E: Metric learning (ours)
```

the included loader works directly.

If it says F, G, or H, the exact `rung_ablation.py` architecture used to train that checkpoint must also be supplied/ported before deployment. Do not substitute another model.

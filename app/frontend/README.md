# Garment Reader — frontend

A static, framework-free frontend with two jobs:

1. **Classify** — upload a garment photo (with or without extra known
   details) and get back predicted `articleType`, `season`, `gender`, and
   `usage`.
2. **Search similar** — upload a photo and get back the 10 most visually
   similar catalog items.

No build step, no framework — plain HTML/CSS/JS. Point it at a backend and
it works.

## Folder structure

```
frontend/
├── index.html            # page structure
├── css/
│   └── styles.css        # all styling (design tokens at the top)
├── js/
│   ├── config.js         # <- backend URLs live here, change this to go live
│   └── app.js            # upload handling, API calls, rendering
└── mock_server/          # fake backend for local dev/demo (see below)
    ├── server.py
    └── requirements.txt
```

## Running it locally (with the mock backend)

The mock server returns realistic-looking fake predictions so you can see
the whole UI working before the real model pipeline is hooked up.

```bash
# 1. start the mock backend
cd frontend/mock_server
pip install -r requirements.txt
python server.py
# -> listening on http://localhost:5000

# 2. in a second terminal, serve the frontend itself (any static server works)
cd ..
python -m http.server 8000
# -> open http://localhost:8000
```

`js/config.js` already points at `http://localhost:5000`, so the two
endpoints above are wired up out of the box.

## Connecting the real backend

Everything the frontend needs to know about your backend lives in **one
file**: `js/config.js`.

```js
const API_CONFIG = {
  classifyUrl: 'https://your-api.example.com/api/classify',
  searchSimilarUrl: 'https://your-api.example.com/api/search-similar',
};
```

Point those two URLs at your real service and nothing else in the frontend
needs to change, as long as your endpoints match the contract below (this
is exactly what `mock_server/server.py` implements — read it alongside this
section if anything is ambiguous).

### `POST /api/classify`

**Request** — `multipart/form-data`:

| field            | required | notes                                   |
|------------------|----------|------------------------------------------|
| `image`          | yes      | the photo file (JPG/PNG)                  |
| `masterCategory` | no       | free text, only sent if the user filled it in |
| `subCategory`    | no       | free text                                 |
| `baseColour`     | no       | free text                                 |
| `year`           | no       | number                                    |

Only the fields the user actually filled in are sent — a blank field is
omitted entirely rather than sent as an empty string, so the backend can
tell "not provided" apart from "provided but blank."

**Response** — `200 OK`, `application/json`:

```json
{
  "articleType": { "label": "Tshirts", "confidence": 0.92 },
  "season":      { "label": "Summer",  "confidence": 0.81 },
  "gender":      { "label": "Women",   "confidence": 0.88 },
  "usage":       { "label": "Casual",  "confidence": 0.95 }
}
```

`confidence` is a 0–1 float; the UI renders it as a percentage and a thin
progress bar. All four keys are expected; if a key is missing that row is
simply skipped.

**Error** — any non-2xx status, `application/json`:

```json
{ "error": "Human-readable explanation shown directly in the UI." }
```

### `POST /api/search-similar`

**Request** — `multipart/form-data`:

| field   | required | notes                     |
|---------|----------|---------------------------|
| `image` | yes      | the photo file (JPG/PNG)  |

**Response** — `200 OK`, `application/json`:

```json
{
  "results": [
    {
      "id": "10001",
      "imageUrl": "https://your-cdn.example.com/images/10001.jpg",
      "similarity": 0.94,
      "articleType": "Tshirts"
    }
  ]
}
```

Return up to 10 items, ordered by descending `similarity` (0–1 float); the
frontend renders at most the first 10 regardless. `articleType` is shown as
a small caption under each thumbnail and can be omitted if you don't have
it.

## Wiring this up to the Task 3 notebook's models

If this frontend is sitting in front of the COSC2753 Task 3 pipeline
(`COSC2753_A2_Task3.ipynb`, Section 14 "Save models & encoders"), a few
things line up directly:

- The saved `manifest.json` / label encoders from Section 14.1 tell you
  exactly which model won for `gender` and `usage`, and what its input
  shape (image size, `META_DIM`, encoder classes) is — a small Flask/FastAPI
  wrapper loading those checkpoints is the fastest path to a real
  `/api/classify` endpoint. `articleType` and `season` come from the
  earlier Task 1 / Task 2 pipeline stages mentioned in that same section.
- The notebook's Section 13 already implements the **missing-metadata
  fallback** logic this frontend's optional fields assume: if the user
  leaves `masterCategory`/`subCategory`/`baseColour`/`year` blank, route
  the request to the image-only model instead of fabricating metadata —
  don't zero-fill a multi-input model's metadata vector, since it was
  never trained on that input.
- For `search-similar`, `imageUrl` needs to resolve to something the
  browser can load directly (a CDN URL, a static file route, or a signed
  URL) — the frontend just renders whatever URL string it's given.

## Design notes

Visual language borrows from garment care labels and spec sheets, since
the tool is reading structured attributes off a product photo: hairline
borders instead of drop-shadowed cards, monospace for the actual predicted
values (the "printed" part of a label), and a warm paper/ink/rust palette
distinct from typical AI-tool defaults. Colors, type, and spacing are
defined as CSS custom properties at the top of `css/styles.css` if you need
to adapt them to a different brand.

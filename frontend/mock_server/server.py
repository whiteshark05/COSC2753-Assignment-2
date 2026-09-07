"""
Mock backend for the Garment Reader frontend.

Implements the exact two endpoints the frontend calls (see ../README.md for
the full contract) using fake but deterministic data, so the UI can be
developed and demoed before the real classification / similarity-search
pipeline is wired up. Swap this out for the real service by pointing
frontend/js/config.js at it -- nothing on the frontend needs to change as
long as the response shapes below are preserved.

Run:
    pip install -r requirements.txt
    python server.py
    # -> listening on http://localhost:5000
"""
import hashlib
import random

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # the frontend is typically opened from a different origin/port

ARTICLE_TYPES = ['Tshirts', 'Shirts', 'Casual Shoes', 'Watches', 'Jeans',
                  'Handbags', 'Sweatshirts', 'Kurtas', 'Sandals', 'Backpacks']
SEASONS  = ['Summer', 'Winter', 'Fall', 'Spring']
GENDERS  = ['Men', 'Women', 'Unisex', 'Boys', 'Girls']
USAGES   = ['Casual', 'Formal', 'Sports', 'Ethnic', 'Party']


def _seed_from_bytes(data: bytes) -> random.Random:
    """Deterministic RNG per image, so re-uploading the same photo gives the
    same demo prediction instead of a different random result each time."""
    digest = hashlib.md5(data).hexdigest()
    return random.Random(int(digest, 16))


@app.post('/api/classify')
def classify():
    file = request.files.get('image')
    if file is None:
        return jsonify(error='No image file was uploaded (expected field "image").'), 400

    rng = _seed_from_bytes(file.read())

    def pick(pool):
        return {'label': rng.choice(pool), 'confidence': round(rng.uniform(0.62, 0.97), 4)}

    response = {
        'articleType': pick(ARTICLE_TYPES),
        'season':      pick(SEASONS),
        'gender':      pick(GENDERS),
        'usage':       pick(USAGES),
        # Demo-only flag showing whether optional metadata was received —
        # a real backend might use this to route to a multi-input vs.
        # image-only model, per the Task 3 pipeline's Stage 1 / Stage 2 split.
        'usedMetadata': any(
            request.form.get(k, '').strip()
            for k in ('masterCategory', 'subCategory', 'baseColour', 'year')
        ),
    }
    return jsonify(response)


@app.post('/api/search-similar')
def search_similar():
    file = request.files.get('image')
    if file is None:
        return jsonify(error='No image file was uploaded (expected field "image").'), 400

    raw = file.read()
    digest = hashlib.md5(raw).hexdigest()
    rng = _seed_from_bytes(raw)

    results = []
    # Descending, slightly-jittered similarity scores so the top result
    # always looks like the strongest match.
    base = 0.97
    for i in range(10):
        base -= rng.uniform(0.015, 0.05)
        results.append({
            'id': f'{digest[:8]}-{i}',
            # picsum.photos serves a stable placeholder image per seed --
            # swap for real catalog image URLs in the real backend.
            'imageUrl': f'https://picsum.photos/seed/{digest[:10]}{i}/300/400',
            'similarity': round(max(base, 0.4), 4),
            'articleType': rng.choice(ARTICLE_TYPES),
        })

    return jsonify(results=results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

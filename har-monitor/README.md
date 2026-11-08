# HAR Monitor

Human Activity Recognition for factory automation. A dashboard and API that classify
wearable sensor windows (UCI HAR, 561 features) into 6 activities:
LAYING, SITTING, STANDING, WALKING, WALKING_UPSTAIRS and WALKING_DOWNSTAIRS.

- **Backend:** FastAPI (Python). The Keras model is exported to NumPy weights, so the deployed app needs only `fastapi` and `numpy`.
- **Frontend:** plain HTML, CSS and JS in `public/` (no build step). The charts are SVG, and the animated background is plain CSS.
- **Hosting:** Vercel. It finds `app.py` automatically, and the files in `public/` are served from its CDN.

```
har-monitor/
├── app.py                   FastAPI app (API + serves public/)
├── requirements.txt         runtime deps (deployed)
├── requirements-dev.txt     + uvicorn, pytest, httpx for local work
├── vercel.json              function settings and excluded files
├── .python-version          Python 3.12 on Vercel
├── model/
│   ├── har_mlp.npz          exported model weights (BatchNorm and scaler folded in)
│   ├── metrics.json         numbers shown on the dashboard
│   └── samples.npz          600 real test windows for the live demo
├── public/
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js, charts.js
├── scripts/export_model.py  Keras → NumPy export (run after training)
├── training/                training code + trained Keras model (not deployed)
└── tests/test_api.py
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000 for the dashboard and http://127.0.0.1:8000/api/docs for the interactive API docs.

Run the tests with `pytest -q`.

## Deploy to Vercel

**Option A: GitHub (recommended)**
1. Push this folder to a GitHub repository.
2. On vercel.com, click **Add New → Project** and import the repository.
3. Leave the framework preset as detected (FastAPI) and click **Deploy**.

**Option B: Vercel CLI**
```bash
npm i -g vercel
vercel          # first time: link or create the project, preview deploy
vercel --prod   # production deploy
```

No environment variables are needed.

## API

| Method | Path | What it does |
|---|---|---|
| GET  | `/api/health` | Status, class names, feature count |
| GET  | `/api/metrics` | Accuracy, F1, confusion matrix and training curves |
| GET  | `/api/features` | The 561 feature names, in order |
| POST | `/api/predict` | `{"features": [561 numbers]}` → activity, confidence, probabilities |
| POST | `/api/predict/batch` | `{"rows": [[561 numbers], …]}` (max 1000 rows) → predictions |
| GET  | `/api/stream?n=1` | Classifies real held-out test windows (the live demo feed) |

Example:
```bash
curl -X POST https://YOUR-APP.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.28, -0.02, ... 561 values ...]}'
```

Predictions under 70% confidence come back with `"low_confidence": true`. The dashboard flags these windows for review instead of triggering a machine action.

## Retrain and re-export the model

```bash
pip install -r training/requirements-train.txt
# put train.csv and test.csv in training/data/
cd training
python rf_baseline.py data        # Random Forest baseline → rf_baseline.joblib
python keras_mlp.py data          # Keras model → best_har_mlp.keras, history.json
cd ..
python scripts/export_model.py --data training/data --artifacts training --history training/history.json
```

The export folds BatchNorm and the StandardScaler into the Dense weights, then checks that the NumPy predictions match Keras before writing `model/`.

## Customise

All settings are at the top of `public/js/app.js` in the `CONFIG` object:
- **Machine response text per activity:** `CONFIG.responses`. The defaults are examples; replace them with your line's real rules.
- **Live feed speed:** `CONFIG.streamEveryMs`.

## Notes and limits

- Vercel limits a request body to 4.5 MB, so the dashboard uploads CSVs in chunks of 250 rows. A full `test.csv` (2,947 rows) works.
- The live demo replays real held-out test windows. To connect real wearables, send their 561-feature vectors to `/api/predict`.
- Accuracy on the held-out test subjects is 93.5% (Keras) and 92.9% (Random Forest). The weakest pair is Sitting and Standing.

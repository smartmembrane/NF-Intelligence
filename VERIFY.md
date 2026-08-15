# Nanofiltration Intelligence Platform - Verification Guide

## What is included

- FastAPI web application with single-recipe prediction, batch screening, monomer screening, SHAP summaries, and model metrics.
- Trained model artifacts, stage-1 characterization models, input data, and archived training outputs.
- Windows one-click launcher, Dockerfile, and API endpoints for independent checks.

## Windows: three-minute verification

1. Extract the ZIP and double-click `start_windows.bat`.
2. Wait for `Uvicorn running on http://0.0.0.0:8000`.
3. Open `http://127.0.0.1:8000/api/health`. Expected result: `status` is `ok` and `two_stage_available` is `true`.
4. Open `http://127.0.0.1:8000`. Keep **two-stage mode** enabled; leave WCA, Zeta potential, pore size, MWCO, and thickness empty; then run one prediction. The result area should identify the characterization source as stage 1.

## API checks

- `GET /api/health` - application and two-stage readiness.
- `GET /api/stage1` - five recipe-to-characterization models and their validation results.
- `GET /api/metrics` - five performance targets, selected models, and group-held-out metrics.
- `GET /api/model_comparison` - every regression candidate ranked by GroupKFold R² and MAE.
- `GET /api/mg_risk` - the parallel Mg²⁺ rejection failure-risk model and validation summary.
- `POST /api/predict` - single-recipe prediction.

## Docker

```bash
docker build -t nf-webapp .
docker run --rm -p 8000:8000 nf-webapp
```

Then open `http://127.0.0.1:8000`.

## Online deployment verification

After Railway or Render gives the service a public URL, replace `YOUR-URL` below:

```text
https://YOUR-URL/api/health
https://YOUR-URL/docs
https://YOUR-URL/
```

The health endpoint must return `status: ok` and `two_stage_available: true` before the platform link is shared with others.

## Reproducibility note

The package includes the dataset and model artifacts used by the application. On a scikit-learn version mismatch, the application retrains from the bundled `data/` files so that local model serialization is compatible with the local environment.

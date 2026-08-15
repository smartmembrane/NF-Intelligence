# Nanofiltration Intelligence Platform

This FastAPI application predicts nanofiltration membrane Mg2+/Li+ separation performance. It supports single-recipe prediction, two-stage recipe-to-characterization-to-performance prediction, monomer screening, SHAP summaries, and downloadable prediction results.

## Publish online

The application must be deployed as one service. Do not deploy `app/static/index.html` by itself: the page calls FastAPI endpoints such as `/api/predict` and `/api/screen`.

### Railway (recommended)

1. Create a private GitHub repository and upload this complete `nf_webapp` folder. Keep `data/` and `models/` because the service needs them at runtime.
2. In Railway, create a project, choose **Deploy from GitHub Repo**, and select the repository.
3. Railway detects the root `Dockerfile`. No build or start command is required.
4. Set the health-check path to `/api/health`, then generate a public domain.
5. After the deployment is healthy, open the generated HTTPS URL.

### Render (alternative)

1. Upload this complete folder to a GitHub repository.
2. In Render, select **New > Web Service**, connect the repository, and choose the **Docker** runtime.
3. Render uses `render.yaml` to check `/api/health`; deploy the service and use the generated `onrender.com` URL.

The container reads the cloud provider's `PORT` environment variable automatically. It falls back to port `8000` only for local Docker use.

## Verify a published deployment

Replace `YOUR-URL` below with the public URL. Do not add a trailing slash.

| Check | URL | Expected result |
|---|---|---|
| Service readiness | `https://YOUR-URL/api/health` | JSON with `"status": "ok"` and `"two_stage_available": true` |
| Web interface | `https://YOUR-URL/` | Nanofiltration Intelligence Platform page |
| API documentation | `https://YOUR-URL/docs` | Interactive FastAPI API documentation |
| Stage-1 evidence | `https://YOUR-URL/api/stage1` | Recipe-to-characterization model information |

Minimum functional check: open the interface, keep two-stage mode enabled, leave characterization fields blank, submit one prediction, and confirm that the result reports stage-1 characterization values.

## Local use

### Windows

Double-click `start_windows.bat`, wait for the server message, then open `http://127.0.0.1:8000`.

### Command line

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t nf-webapp .
docker run --rm -p 8000:8000 nf-webapp
```

Then open `http://127.0.0.1:8000`.

## Reproducibility package

For a research release, publish the following together:

- Online platform URL and the `/api/health` verification URL.
- Versioned source code, `requirements.txt`, Dockerfile, input data, and model artifacts.
- `VERIFY.md`, one example input, its expected output, and the release date/version.
- A short statement of model scope: screening prioritizes candidate recipes; it does not replace experimental validation.

The application checks whether the bundled model can be loaded in the current scikit-learn environment. If it cannot, it retrains from the bundled `data/` files.

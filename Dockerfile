FROM python:3.11-slim

WORKDIR /code
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY training ./training
COPY models ./models
COPY data ./data
RUN mkdir -p results

EXPOSE 8000
# 首次启动会自动训练模型（约1-2分钟），之后直接加载
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

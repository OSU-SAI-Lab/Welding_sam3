FROM python:3.12-slim

# The dataset exporter re-extracts frames from the source video, so this image
# needs ffmpeg as well as the Python runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Runs as the same uid as the video service so both can use the shared PVC.
USER 1000

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

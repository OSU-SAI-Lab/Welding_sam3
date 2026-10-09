# The dataset exporter shells out to ffmpeg to re-extract frames from a source
# video, so this image needs that one binary — and nothing else from the
# multimedia stack.
#
# Installing Debian's `ffmpeg` package pulls in the whole desktop media tree
# (libdav1d, libcodec2, chromaprint, libdc1394 and ~100 more) for several
# hundred megabytes, on an image that is otherwise a small HTTP proxy. NRP pulls
# this on every rollout, so copying a static build keeps it to one file.
#
# To drop the third-party dependency instead, replace the COPY below with:
#   RUN apt-get update \
#    && apt-get install -y --no-install-recommends ffmpeg \
#    && rm -rf /var/lib/apt/lists/*
FROM mwader/static-ffmpeg:7.1 AS ffmpeg

FROM python:3.12-slim

COPY --from=ffmpeg /ffmpeg /usr/local/bin/ffmpeg

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Runs as the same uid as the video service so both can use the shared PVC.
USER 1000

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

"""SAM3 video GPU service entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import logs, track_jobs
from app.config import CORS_ALLOW_ORIGINS, HOST, PORT

logs.configure()
from app.routes import router
from app.track_routes import router as track_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Before serving: jobs a previous run left active have no worker anymore.
    track_jobs.recover_interrupted()
    yield


app = FastAPI(title="SAM3 Video Service", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    # Wildcard origins + credentials is rejected by browsers; only allow
    # credentials when a concrete allow-list is configured.
    allow_credentials=CORS_ALLOW_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.include_router(track_router)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_config=logs.uvicorn_log_config())

"""SAM3 video GPU service entrypoint."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import HOST, PORT
from app.routes import router
from app.track_routes import router as track_router

app = FastAPI(title="SAM3 Video Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.include_router(track_router)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import setting
from backend.api import endpoints

app = FastAPI(description="REST API for Image Generation", docs_url="/docs", redoc_url="/redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(endpoints.router, prefix=setting.PREFIX_API, tags=["Image Generation"])

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "ok",
        "service": setting.PROJECT_NAME
    }

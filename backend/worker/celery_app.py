from celery import Celery
from backend.core.config import setting

celery_app = Celery(
    "text2image_worker",
    broker=setting.REDIS_URL,
    backend=setting.REDIS_URL,
    include=['backend.worker.tasks']
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)
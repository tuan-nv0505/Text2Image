import json

from fastapi import APIRouter
from celery.result import AsyncResult
from starlette.responses import StreamingResponse
import redis

from backend.core.config import setting
from backend.schemas.generate import GenerateRequest, TaskResponse
from backend.worker.tasks import generate_image_task

router = APIRouter()

@router.post("/generate", response_model=TaskResponse)
async def create_generation_task(request: GenerateRequest):
    task = generate_image_task.delay(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        seed=request.seed
    )
    return TaskResponse(task_id=task.id, status="processing")


@router.get("/stream/{task_id}")
async def stream_task_status(task_id: str):

    async def event_generator():
        task_result = AsyncResult(task_id)
        if task_result.state == "SUCCESS":
            yield f"data: {json.dumps({'status': 'completed', 'image_url': task_result.result.get('image_url')})}\n\n"
            return
        elif task_result.state == "FAILED":
            yield f"data: {json.dumps({'status': 'failed', 'error': str(task_result.info)})}\n\n"
            return

        redis_conn = redis.asyncio.from_url(setting.REDIS_URL)
        pubsub = redis_conn.pubsub()
        channel_name = f"task_events:{task_id}"

        await pubsub.subscribe(channel_name)
        try:
            yield f"data: {json.dumps({'status': 'processing'})}\n\n"
            async for message in pubsub.listen():
                if message["type"] == "message":
                    raw_json_data = message["data"].decode("utf-8")
                    yield f"data: {raw_json_data}\n\n"
                    break
        finally:
            await pubsub.unsubscribe(channel_name)
            await pubsub.close()
            await redis_conn.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
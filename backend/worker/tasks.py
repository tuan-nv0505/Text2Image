from io import BytesIO
import json
import redis
from celery.signals import worker_process_init

from backend.worker.celery_app import celery_app
from backend.core.config import setting
from backend.services.s3_service import upload_image, download_checkpoint

from ai_engine.registry.text2image_pipeline import TextToImagePipeline 

from utils.logger import logger

pipeline = None

redis_client = redis.from_url(setting.REDIS_URL)

@worker_process_init.connect
def initialize_model(**kwargs):
    global pipeline
    logger.info("Initializing TextToImagePipeline...")

    download_checkpoint(setting.S3_PREFIX_CHECKPOINT, setting.DIT_CHECKPOINT_PATH)
    
    pipeline = TextToImagePipeline(
        dit_model_name=setting.DIT_MODEL_NAME,
        dit_checkpoint_path=setting.DIT_CHECKPOINT_PATH,
        vae_name=setting.VAE_NAME,
        t5_name=setting.T5_NAME,
        latent_size=setting.LATENT_SIZE
    )
    logger.info("Pipeline initialized successfully.")

@celery_app.task(bind=True, soft_time_limit=300, time_limit=330, max_retries=2)
def generate_image_task(self, prompt: str, negative_prompt: str, num_inference_steps: int, guidance_scale: float, seed: int | None):
    global pipeline
    if pipeline is None:
        raise RuntimeError("Model pipeline is not initialized. Ensure that the worker process has been properly set up.")

    task_id = self.request.id
    channel_name = f"task_events:{task_id}"

    try:
        image = pipeline.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        
        image_url = upload_image(buffered)

        result_payload = {"status": "completed", "image_url": image_url}
        redis_client.publish(channel_name, json.dumps(result_payload))

        return result_payload

    except Exception as e:
        logger.exception("Error occurred while executing the task!")
        error_payload = {"status": "failed", "error": str(e)}
        redis_client.publish(channel_name, json.dumps(error_payload))

        self.update_state(state="FAILED", meta={"error": str(e)})
        raise self.retry(exc=e, countdown=5)
from pydantic_settings import BaseSettings

class Setting(BaseSettings):
    PROJECT_NAME: str
    PREFIX_API: str

    DIT_MODEL_NAME: str
    DIT_CHECKPOINT_PATH: str
    VAE_NAME: str
    T5_NAME: str
    LATENT_SIZE: int
    
    REDIS_URL: str
    
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET_NAME: str
    S3_PREFIX_CHECKPOINT: str
    S3_PREFIX_IMAGE: str
    
    class Config:
        env_file = ".env"

setting = Setting()
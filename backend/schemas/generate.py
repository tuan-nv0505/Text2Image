from pydantic import BaseModel, Field
from typing import Optional

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: Optional[str] = Field("",)
    num_inference_steps: int = Field(250, ge=1, le=1000)
    guidance_scale: float = Field(2.0, ge=1.0, le=10.0)
    seed: Optional[int] = Field(None)

class TaskResponse(BaseModel):
    task_id: str
    status: str
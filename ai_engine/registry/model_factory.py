from ai_engine.models.diffusion_transformer import DiffusionTransformer_models
from ai_engine.models.vae import VAE
from ai_engine.diffusion import create_diffusion
from ai_engine.models.t5 import T5Embedder

from .checkpoint_loader import CheckpointLoader


class ModelFactory:
    @staticmethod
    def create_dit(*, model_name: str, checkpoint_path: str, latent_size: int, device: str = "cuda", use_ema: bool = True):
        model = DiffusionTransformer_models[model_name](input_size=latent_size)
        checkpoint = CheckpointLoader.load(checkpoint_path)

        if use_ema and "ema" in checkpoint:
            state_dict = checkpoint["ema"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)

        return model

    @staticmethod
    def create_vae(*, pretrained_name: str, device: str = "cuda"):
        return VAE(vae=pretrained_name, device=device)

    @staticmethod
    def create_diffusion(*, sampling_steps: int):
        return create_diffusion(str(sampling_steps))

    @staticmethod
    def create_t5(*, dir_or_name: str = 'google/t5-v1_1-large', device: str = "cpu", model_max_length: int = 120):
        return T5Embedder(
            device=device,
            dir_or_name=dir_or_name,
            model_max_length=model_max_length
        )
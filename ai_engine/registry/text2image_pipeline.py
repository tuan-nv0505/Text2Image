import time
from typing import Optional

import torch
import numpy as np
from PIL import Image

from ai_engine.registry.model_registry import ModelRegistry

from utils.logger import logger


class TextToImagePipeline:
    def __init__(
            self,
            dit_model_name: str = "DiT-S/2",
            dit_checkpoint_path: str = "",
            vae_name: str = "ema",
            t5_name: str = "google/t5-v1_1-large",
            latent_size: int = 32,
    ):
        self.dit_model_name = dit_model_name
        self.dit_checkpoint_path = dit_checkpoint_path
        self.vae_name = vae_name
        self.t5_name = t5_name
        self.latent_size = latent_size

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.amp_dtype = torch.float32
        else:
            self.device = torch.device("cpu")
            self.amp_dtype = torch.float32

        logger.info(f"Pipeline initialized on device: {self.device}, Dtype: {self.amp_dtype}")

        self.registry = ModelRegistry()

    @torch.inference_mode()
    def generate(
            self,
            prompt: str,
            negative_prompt: str = "",
            num_inference_steps: int = 250,
            guidance_scale: float = 4.0,
            seed: Optional[int] = None
    ) -> Image.Image:

        start_time = time.time()

        try:
            if seed is not None:
                torch.manual_seed(seed)
                if self.device.type == 'cuda':
                    torch.cuda.manual_seed_all(seed)

            logger.info(f"Starting image generation. Prompt: '{prompt[:50]}...'")

            logger.debug("Running T5 Text Encoder...")
            t5 = self.registry.get_t5(dir_or_name=self.t5_name, device=self.device)

            cond_embeddings, cond_mask = t5.get_text_embeddings([prompt])
            uncond_embeddings, uncond_mask = t5.get_text_embeddings([negative_prompt])

            context = torch.cat([cond_embeddings, uncond_embeddings], dim=0).unsqueeze(1)
            mask = torch.cat([cond_mask, uncond_mask], dim=0)

            t5.to("cpu")

            logger.debug("T5 successfully offloaded to CPU.")

            logger.info(f"Running Diffusion loop for {num_inference_steps} steps...")

            z = torch.randn(1, 4, self.latent_size, self.latent_size, device=self.device)
            z = torch.cat([z, z], dim=0)

            dit = self.registry.get_dit(
                model_name=self.dit_model_name,
                checkpoint_path=self.dit_checkpoint_path,
                latent_size=self.latent_size,
                device=self.device
            )
            diffusion = self.registry.get_diffusion(sampling_steps=num_inference_steps)

            autocast_device = "cuda" if self.device.type == "cuda" else "cpu"

            with torch.autocast(device_type=autocast_device, dtype=self.amp_dtype):
                model_kwargs = dict(y=context, mask=mask, cfg_scale=guidance_scale)
                samples = diffusion.p_sample_loop(
                    dit.forward_with_cfg,
                    z.shape,
                    z,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=False,
                    device=self.device
                )

            samples, _ = samples.chunk(2, dim=0)
            dit.to("cpu")
            if self.device.type == "cuda": torch.cuda.empty_cache()
            logger.debug("DiT successfully offloaded to CPU.")

            logger.info("Decoding latent to image using VAE...")
            vae = self.registry.get_vae(pretrained_name=self.vae_name, device=self.device)

            with torch.autocast(device_type=autocast_device, dtype=self.amp_dtype):
                decoded_samples = vae.decode(latent=samples)
                decoded_samples = (decoded_samples / 2 + 0.5).clamp(0, 1)

            vae.to("cpu")
            if self.device.type == "cuda": torch.cuda.empty_cache()
            logger.debug("VAE successfully offloaded to CPU.")

            image = self._tensor_to_pil(decoded_samples)

            process_time = time.time() - start_time
            logger.info(f"Image generation completed successfully in {process_time:.2f} seconds.")

            return image

        except Exception as e:
            logger.exception("Critical error during image generation!")
            self.registry.clear_cache()
            raise e

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        tensor = tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (tensor * 255).round().astype(np.uint8)
        return Image.fromarray(images[0])
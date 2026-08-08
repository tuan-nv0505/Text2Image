import torch
from diffusers.models import AutoencoderKL


class VAE:
    def __init__(self, vae, device):
        self.model = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae}").to(device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, image: torch.Tensor):
        latent = self.model.encode(image).latent_dist.sample()
        latent = latent * 0.18215
        return latent

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor):
        latent = latent / 0.18215
        image = self.model.decode(latent).sample
        return image

    def to(self, device):
        self.model.to(device)
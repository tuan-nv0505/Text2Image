from typing import Dict
import threading
import torch

from .model_factory import ModelFactory


class ModelRegistry:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        self._models: Dict[str, object] = {}
        self._vaes: Dict[str, object] = {}
        self._diffusions: Dict[int, object] = {}
        self._t5s: Dict[str, object] = {}

    def get_dit(self, model_name: str, checkpoint_path: str, latent_size: int, device: str = "cuda",
                use_ema: bool = True):
        cache_key = f"{model_name}_{checkpoint_path}_{use_ema}"

        with self._lock:
            if cache_key not in self._models:
                self._models[cache_key] = ModelFactory.create_dit(
                    model_name=model_name,
                    checkpoint_path=checkpoint_path,
                    latent_size=latent_size,
                    device=device,
                    use_ema=use_ema
                )
            else:
                self._models[cache_key].to(device)

        return self._models[cache_key]

    def get_vae(self, pretrained_name: str, device: str = "cuda"):
        cache_key = pretrained_name

        with self._lock:
            if cache_key not in self._vaes:
                self._vaes[cache_key] = ModelFactory.create_vae(
                    pretrained_name=pretrained_name,
                    device=device
                )
            else:
                self._vaes[cache_key].to(device)

        return self._vaes[cache_key]

    def get_diffusion(self, sampling_steps: int):
        with self._lock:
            if sampling_steps not in self._diffusions:
                self._diffusions[sampling_steps] = ModelFactory.create_diffusion(
                    sampling_steps=sampling_steps
                )
        return self._diffusions[sampling_steps]

    def get_t5(self, dir_or_name: str = 'google/t5-v1_1-large', device: str = "cpu", model_max_length: int = 120):
        cache_key = f"{dir_or_name}_{model_max_length}"

        with self._lock:
            if cache_key not in self._t5s:
                self._t5s[cache_key] = ModelFactory.create_t5(
                    dir_or_name=dir_or_name,
                    device=device,
                    model_max_length=model_max_length
                )
            else:
                self._t5s[cache_key].to(device)

        return self._t5s[cache_key]

    def clear_cache(self):
        with self._lock:
            self._models.clear()
            self._vaes.clear()
            self._diffusions.clear()
            self._t5s.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
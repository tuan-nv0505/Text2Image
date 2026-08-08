import logging
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple

from ai_engine.registry.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class TextToImagePipeline:
    def __init__(
            self,
            dit_model_name: str = "DiT-S/2",
            dit_checkpoint_path: str = "",
            vae_name: str = "ema",
            t5_name: str = "google/t5-v1_1-large",
            latent_size: int = 32,  # Ví dụ: Ảnh 256x256 -> latent 32
            device: str = "cuda"
    ):
        self.dit_model_name = dit_model_name
        self.dit_checkpoint_path = dit_checkpoint_path
        self.vae_name = vae_name
        self.t5_name = t5_name
        self.latent_size = latent_size

        # Mặc định thiết bị tính toán chính
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Gọi Registry (Thủ thư)
        self.registry = ModelRegistry()

        # Kiểm tra hỗ trợ Mixed Precision để tăng tốc
        self.amp_dtype = torch.bfloat16 if (
                    self.device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

    @torch.inference_mode()
    def generate(
            self,
            prompt: str,
            negative_prompt: str = "",
            num_inference_steps: int = 250,
            guidance_scale: float = 4.0,
            seed: Optional[int] = None
    ) -> Image.Image:
        """
        Quy trình sinh ảnh từ văn bản End-to-End.
        """
        # 1. Cài đặt seed để có thể tái tạo lại ảnh (nếu cần)
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)

        logger.info(f"Bắt đầu sinh ảnh với prompt: '{prompt}'")

        # ==========================================
        # BƯỚC 1: XỬ LÝ VĂN BẢN BẰNG T5
        # ==========================================
        t5 = self.registry.get_t5(dir_or_name=self.t5_name, device=self.device)

        # Lấy embedding cho prompt (có điều kiện) và negative prompt (không điều kiện)
        cond_embeddings, cond_mask = t5.get_text_embeddings([prompt])
        uncond_embeddings, uncond_mask = t5.get_text_embeddings([negative_prompt])

        # Ghép lại để chạy Classifier-Free Guidance (CFG)
        context = torch.cat([cond_embeddings, uncond_embeddings], dim=0).unsqueeze(1)
        mask = torch.cat([cond_mask, uncond_mask], dim=0)

        # QUAN TRỌNG: Offload T5 về CPU ngay lập tức
        t5.offload()
        logger.debug("Đã offload T5 về CPU.")

        # ==========================================
        # BƯỚC 2: QUÁ TRÌNH KHỬ NHIỄU (DIFFUSION) BẰNG DiT
        # ==========================================
        # Khởi tạo nhiễu ngẫu nhiên (Latent noise)
        z = torch.randn(1, 4, self.latent_size, self.latent_size, device=self.device)
        z = torch.cat([z, z], dim=0)  # Ghép đôi cho CFG

        # Gọi DiT và thuật toán lấy mẫu (Diffusion) từ Registry
        dit = self.registry.get_dit(
            model_name=self.dit_model_name,
            checkpoint_path=self.dit_checkpoint_path,
            latent_size=self.latent_size,
            device=self.device
        )
        diffusion = self.registry.get_diffusion(sampling_steps=num_inference_steps)

        logger.info(f"Đang chạy khử nhiễu {num_inference_steps} steps...")

        # Chạy vòng lặp khử nhiễu (Denoising loop) với Mixed Precision
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
            model_kwargs = dict(y=context, mask=mask, cfg_scale=guidance_scale)
            samples = diffusion.p_sample_loop(
                dit.forward_with_cfg,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,  # Có thể tắt nếu chạy nền trên server
                device=self.device
            )

        # Tách phần có điều kiện (conditional) ra sau khi dùng CFG
        samples, _ = samples.chunk(2, dim=0)

        # QUAN TRỌNG: Offload DiT về CPU
        dit.to("cpu")
        logger.debug("Đã offload DiT về CPU.")

        # ==========================================
        # BƯỚC 3: GIẢI MÃ LATENT THÀNH ẢNH BẰNG VAE
        # ==========================================
        vae = self.registry.get_vae(pretrained_name=self.vae_name, device=self.device)

        logger.info("Đang giải mã ảnh (VAE decoding)...")
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
            decoded_samples = vae.decode(latent=samples)
            # Chuẩn hóa tensor từ khoảng [-1, 1] về [0, 1]
            decoded_samples = (decoded_samples / 2 + 0.5).clamp(0, 1)

        # QUAN TRỌNG: Offload VAE về CPU
        vae.to("cpu")
        logger.debug("Đã offload VAE về CPU.")

        # ==========================================
        # BƯỚC 4: HẬU XỬ LÝ ẢNH (POST-PROCESSING)
        # ==========================================
        image = self._tensor_to_pil(decoded_samples)
        logger.info("Hoàn thành sinh ảnh!")

        return image

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Chuyển đổi tensor PyTorch thành ảnh PIL để có thể lưu dưới dạng PNG/JPG."""
        # Chuyển [B, C, H, W] -> [B, H, W, C] và gỡ khỏi GPU
        tensor = tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
        # Scale về [0, 255]
        images = (tensor * 255).round().astype(np.uint8)
        # Vì batch_size hiện tại là 1, ta lấy ảnh đầu tiên
        return Image.fromarray(images[0])
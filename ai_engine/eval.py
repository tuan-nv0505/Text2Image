import os
import argparse
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from tqdm import tqdm

# Import từ thư viện của bạn
from ai_engine.diffusion import create_diffusion
from ai_engine.models.diffusion_transformer import DiffusionTransformer_models
from ai_engine.models.t5 import T5Embedder
from ai_engine.models.vae import VAE

# Import các độ đo từ torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# --- TẠO MỘT DATASET ĐÁNH GIÁ ĐƠN GIẢN ---
# Chú ý: Bạn cần thay thế phần này bằng logic đọc file ảnh/text thực tế của bạn
class Flickr8kEvalDataset(Dataset):
    def __init__(self, image_dir, text_file, image_size):
        """
        image_dir: Thư mục chứa ảnh gốc Flickr8k
        text_file: File txt/csv chứa mapping từ tên ảnh sang caption
        """
        # Load danh sách ảnh và caption tương ứng ở đây
        self.data = []  # List of dict: [{'image_path': '...', 'caption': '...'}, ...]
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),  # Trả về range [0, 1]
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        from PIL import Image
        img = Image.open(item['image_path']).convert('RGB')
        img_tensor = self.transform(img)
        # Đưa ảnh về kiểu uint8 [0, 255] cho FID tính toán chuẩn nhất
        img_uint8 = (img_tensor * 255).to(torch.uint8)
        return img_uint8, item['caption']


# --- HÀM CHÍNH ---
def main(args, ckpt_path, num_samples):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # 1. Khởi tạo mô hình
    latent_size = args.dataset.image_size // 8
    model = DiffusionTransformer_models[args.model.name](input_size=latent_size).to(device)

    # RẤT QUAN TRỌNG: Sử dụng trọng số của EMA model để đánh giá (sẽ cho chất lượng ảnh tốt hơn nhiều)
    model.load_state_dict(checkpoint["ema"])
    model.eval()

    diffusion = create_diffusion(timestep_respacing=str(args.sampling.num_sampling_steps))
    vae = VAE(args.model.vae, device)
    embedder = T5Embedder(device=device)

    # 2. Khởi tạo Metrics
    # FID cần normalize=False vì ta sẽ truyền ảnh tensor dạng uint8 (0-255)
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    # CLIP Score mặc định sử dụng model của OpenAI
    clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)

    # 3. Khởi tạo Dataloader
    dataset = Flickr8kEvalDataset(
        image_dir="path/to/raw/images",  # TODO: Đổi thành đường dẫn thật
        text_file="path/to/captions",  # TODO: Đổi thành đường dẫn thật
        image_size=args.dataset.image_size
    )
    # Batch size nên nhỏ vì quá trình sinh ảnh tốn nhiều VRAM
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

    uncond_embeddings, uncond_mask = embedder.get_text_embeddings([""])

    print("Bắt đầu sinh ảnh và tính toán metrics...")
    samples_processed = 0

    with torch.inference_mode():
        for real_images, prompts in tqdm(dataloader, total=num_samples // 4):
            if samples_processed >= num_samples:
                break

            real_images = real_images.to(device)  # shape: [B, 3, H, W], uint8
            batch_size = real_images.shape[0]

            # --- Sinh ảnh (Generation) ---
            # Extract Text Embeddings
            cond_embeddings, cond_mask = embedder.get_text_embeddings(prompts)

            # Khớp shape cho Classifier-Free Guidance (CFG)
            # Nhân bản uncond_embeddings cho bằng batch_size
            uncond_emb_batch = uncond_embeddings.expand(batch_size, -1, -1)
            uncond_mask_batch = uncond_mask.expand(batch_size, -1)

            fixed_embeddings = torch.cat([cond_embeddings, uncond_emb_batch], dim=0).unsqueeze(1)
            fixed_attention_mask = torch.cat([cond_mask, uncond_mask_batch], dim=0)

            # Tạo noise ban đầu
            z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
            z_cfg = torch.cat([z, z], dim=0)

            model_kwargs = dict(
                y=fixed_embeddings,
                mask=fixed_attention_mask,
                cfg_scale=args.sampling.cfg_scale,
            )

            # Denoising loop
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg,
                z_cfg.shape,
                z_cfg,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=device,
            )

            # Tách lấy phần cond (nửa đầu của tensor sau khi đã dùng CFG)
            samples, _ = samples.chunk(2, dim=0)

            # Decode bằng VAE
            fake_images = vae.decode(latent=samples)

            # Xử lý tensor ảnh sinh ra từ [-1, 1] về [0, 255] kiểu uint8
            fake_images = (fake_images / 2 + 0.5).clamp(0, 1)
            fake_images_uint8 = (fake_images * 255).to(torch.uint8)

            # --- Cập nhật Metrics ---
            # FID cần cập nhật cả ảnh thật và ảnh giả (để so sánh phân phối)
            fid_metric.update(real_images, real=True)
            fid_metric.update(fake_images_uint8, real=False)

            # CLIP Score cần ảnh giả và text prompt tương ứng
            clip_metric.update(fake_images_uint8, prompts)

            samples_processed += batch_size

    # 4. Tính toán kết quả cuối cùng
    print("\nĐang tính toán điểm số (có thể mất vài phút cho FID)...")
    fid_score = fid_metric.compute()
    clip_score = clip_metric.compute()

    print("=" * 40)
    print(f"Số lượng ảnh đánh giá (Samples): {samples_processed}")
    print(f"FID Score (càng thấp càng tốt): {fid_score.item():.4f}")
    print(f"CLIP Score (càng cao càng tốt): {clip_score.item():.4f}")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint.pt")
    parser.add_argument("--num_samples", type=int, default=1000, help="Số lượng ảnh để test")
    cmd_args = parser.parse_args()

    args = OmegaConf.load(cmd_args.config)
    main(args, cmd_args.ckpt, cmd_args.num_samples)
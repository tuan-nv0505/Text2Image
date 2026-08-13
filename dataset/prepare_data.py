import os
import gc
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from dataset.utils import center_crop_arr
from ai_engine.models.vae import VAE
from ai_engine.models.t5 import T5Embedder


def extract_features_flickr8k(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.latents_dir, exist_ok=True)
    os.makedirs(args.text_embs_dir, exist_ok=True)

    image_dir = os.path.join(args.data_path, "Flicker8k_Dataset")
    caption_path = os.path.join(args.data_path, "Flickr8k_text/Flickr8k.token.txt")

    images_list = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    latents_list = [f for f in os.listdir(args.latents_dir) if f.endswith('.pt')]

    need_latents = len(latents_list) < len(images_list)
    need_text = not os.path.exists(os.path.join(args.text_embs_dir, "emb_000000.pt"))

    if not need_latents and not need_text:
        return

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    with torch.no_grad():
        if need_latents:
            vae = VAE(args.vae, device)

            for img_name in tqdm(images_list, desc="Extracting Latents"):
                save_path = os.path.join(args.latents_dir, f"{img_name}.pt")
                if os.path.exists(save_path): continue

                img_path = os.path.join(image_dir, img_name)
                pil_image = Image.open(img_path).convert("RGB")
                img_tensor = transform(pil_image).unsqueeze(0).to(device)

                latent = vae.encode(img_tensor)
                torch.save(latent.squeeze(0).cpu(), save_path)

            del vae
            gc.collect()
            torch.cuda.empty_cache()

        if need_text:
            embedder = T5Embedder(device=device)

            captions = []
            with open(caption_path, "r") as f:
                for line in f:
                    if line.strip():
                        captions.append(line.strip().split("\t")[1])

            for idx, caption in enumerate(tqdm(captions, desc="Extracting Text")):
                save_path = os.path.join(args.text_embs_dir, f"emb_{idx:06d}.pt")
                if os.path.exists(save_path): continue

                embeddings, attention_mask = embedder.get_text_embeddings([caption])
                torch.save({
                    'embeddings': embeddings[0].cpu().to(torch.bfloat16),
                    'attention_mask': attention_mask[0].cpu()
                }, save_path)

            del embedder
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--latents-dir", type=str, required=True)
    parser.add_argument("--text-embs-dir", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--vae", type=str, default="ema")
    args = parser.parse_args()

    extract_features_flickr8k(args)
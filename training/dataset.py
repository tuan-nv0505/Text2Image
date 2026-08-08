import os
import gc
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.distributed as dist

from training.utils import center_crop_arr
from ai_engine.models.vae import VAE


class Flickr8kDataset(Dataset):
    def __init__(
            self,
            root,
            latents_dir,
            text_embs_dir,
            vae="ema",
            split="train",
            image_size=256,
            device=None,
            transform=None
    ):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.vae = vae
        self.latents_dir = latents_dir
        self.text_embs_dir = text_embs_dir

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.is_dist = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_dist else 0

        if self.rank == 0:
            self._prepare_data_if_needed()

        if self.is_dist:
            dist.barrier()

        self.samples = []
        self._load_samples()

    def _prepare_data_if_needed(self):
        os.makedirs(self.latents_dir, exist_ok=True)
        os.makedirs(self.text_embs_dir, exist_ok=True)

        image_dir = os.path.join(self.root, "Flicker8k_Dataset")
        caption_path = os.path.join(self.root, "Flickr8k_text/Flickr8k.token.txt")

        images_list = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        latents_list = [f for f in os.listdir(self.latents_dir) if f.endswith('.pt')]

        need_latents = len(latents_list) < len(images_list)
        need_text = not os.path.exists(os.path.join(self.text_embs_dir, "emb_000000.pt"))

        if not need_latents and not need_text:
            return

        with torch.no_grad():
            if need_latents:
                vae = VAE(self.vae, self.device)

                for img_name in tqdm(images_list, desc="Extracting Latents"):
                    save_path = os.path.join(self.latents_dir, f"{img_name}.pt")
                    if os.path.exists(save_path): continue

                    img_path = os.path.join(image_dir, img_name)
                    img_tensor = self.transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(self.device)
                    latent = vae.encode(img_tensor)
                    torch.save(latent.squeeze(0).cpu(), save_path)
                del vae

            if need_text:
                from ai_engine.models.t5 import T5Embedder
                embedder = T5Embedder(device=self.device)

                captions = []
                with open(caption_path, "r") as f:
                    for line in f:
                        if line.strip(): captions.append(line.strip().split("\t")[1])

                for idx, caption in enumerate(tqdm(captions, desc="Extracting Text")):
                    save_path = os.path.join(self.text_embs_dir, f"emb_{idx:06d}.pt")
                    if os.path.exists(save_path): continue

                    embeddings, attention_mask = embedder.get_text_embeddings([caption])
                    torch.save({
                        'embeddings': embeddings[0].cpu().to(torch.bfloat16),
                        'attention_mask': attention_mask[0].cpu()
                    }, save_path)
                del embedder

        gc.collect()
        torch.cuda.empty_cache()

    def _load_samples(self):
        split_files = {
            "train": "Flickr8k_text/Flickr_8k.trainImages.txt",
            "val": "Flickr8k_text/Flickr_8k.devImages.txt",
            "test": "Flickr8k_text/Flickr_8k.testImages.txt"
        }
        split_path = os.path.join(self.root, split_files[self.split])
        with open(split_path, "r") as f:
            valid_images = {line.strip() for line in f}

        caption_path = os.path.join(self.root, "Flickr8k_text/Flickr8k.token.txt")
        idx = 0
        with open(caption_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                image_name = line.split("\t")[0].split("#")[0]
                if image_name in valid_images:
                    self.samples.append((image_name, idx))
                idx += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, caption_idx = self.samples[idx]

        latent = torch.load(os.path.join(self.latents_dir, f"{image_name}.pt"), weights_only=True)
        text_data = torch.load(os.path.join(self.text_embs_dir, f"emb_{caption_idx:06d}.pt"), weights_only=True)

        return latent, text_data['embeddings'], text_data['attention_mask']


if __name__ == "__main__":
    import argparse
    from torchvision import transforms


    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--vae", type=str, default="ema")
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = Flickr8kDataset(
        root=args.data_path,
        vae=args.vae,
        image_size=args.image_size,
        transform=transform
    )
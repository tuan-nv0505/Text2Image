import os
import torch
import random
from collections import defaultdict
from torch.utils.data import Dataset


class Flickr8kDataset(Dataset):
    def __init__(self, root, latents_dir, text_embs_dir, split="train", p_uncond=0.1):
        self.root = root
        self.split = split
        self.latents_dir = latents_dir
        self.text_embs_dir = text_embs_dir
        self.p_uncond = p_uncond

        self.samples = []
        self._load_samples()

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
        image_to_captions = defaultdict(list)
        idx = 0

        with open(caption_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue

                image_name = line.split("\t")[0].split("#")[0]
                if image_name in valid_images:
                    image_to_captions[image_name].append(idx)
                idx += 1

        self.samples = []
        for img, caps in image_to_captions.items():
            for cap_idx in caps:
                self.samples.append((img, cap_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, caption_idx = self.samples[idx]

        latent_path = os.path.join(self.latents_dir, f"{image_name}.pt")
        latent = torch.load(latent_path, weights_only=True)

        text_path = os.path.join(self.text_embs_dir, f"emb_{caption_idx:06d}.pt")
        text_data = torch.load(text_path, weights_only=True)

        embeddings = text_data['embeddings']
        attention_mask = text_data['attention_mask']

        if random.random() < self.p_uncond:
            embeddings = torch.zeros_like(embeddings)
            attention_mask = torch.zeros_like(attention_mask)

        return latent, embeddings, attention_mask
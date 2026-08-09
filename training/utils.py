import logging
import os
import sys
from collections import OrderedDict
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch import distributed as dist


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    dist.destroy_process_group()


def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            force=True,
            format='[\033[32m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f"{logging_dir}/log.txt")
            ]
        )
        logger = logging.getLogger(__name__)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("diffusers").setLevel(logging.ERROR)
    else:
        logger = logging.getLogger(__name__)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(logging.NullHandler())

    return logger


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def manage_checkpoints(checkpoint_dir, max_to_keep_checkpoint=10):
    checkpoints = sorted(glob(os.path.join(checkpoint_dir, "checkpoint_*.pt")), key=os.path.getmtime)
    checkpoints = [c for c in checkpoints if "checkpoint_latest.pt" not in c]
    while len(checkpoints) > max_to_keep_checkpoint:
        oldest_checkpoint = checkpoints.pop(0)
        try:
            if os.path.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)
        except OSError as e:
            print(f"Error deleting old checkpoint {oldest_checkpoint}: {e}")

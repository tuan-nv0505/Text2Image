import logging
import os
import sys
from collections import OrderedDict
from glob import glob

import torch
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


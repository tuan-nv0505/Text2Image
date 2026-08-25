import os
from dotenv import load_dotenv
load_dotenv()

import argparse
from copy import deepcopy
from glob import glob
from time import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
from omegaconf import OmegaConf

from ai_engine.diffusion import create_diffusion
from ai_engine.models.diffusion_transformer import DiffusionTransformer_models
from ai_engine.models.t5 import T5Embedder
from ai_engine.models.vae import VAE
from dataset.dataset import Flickr8kDataset
from training.utils import cleanup, create_logger, requires_grad, update_ema
from training.manage_checkpoint import CheckpointManager

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    device = local_rank
    torch.cuda.set_device(device)

    assert (args.training.global_batch_size % world_size == 0), "Batch size must be divisible by world size."
    batch_size_per_gpu = int(args.training.global_batch_size // world_size)

    seed = args.training.global_seed * world_size + rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    assert (args.dataset.image_size % 8 == 0), "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.dataset.image_size // 8

    if rank == 0:
        os.makedirs(args.logging.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.logging.results_dir}/*"))
        model_string_name = args.model.name.replace("/", "-")
        experiment_dir = f"{args.logging.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        with open(f"{experiment_dir}/config.yaml", "w") as f:
            OmegaConf.save(config=args, f=f)
            logger.info("Saved config.yaml to experiment directory.")

        vae = VAE(args.model.vae, "cpu")
        embedder = T5Embedder(device=device)

        cond_embeddings, cond_mask = embedder.get_text_embeddings([args.sampling.prompt])
        uncond_embeddings, uncond_mask = embedder.get_text_embeddings([""])
        fixed_embeddings = torch.cat([cond_embeddings, uncond_embeddings], dim=0).unsqueeze(1)
        fixed_attention_mask = torch.cat([cond_mask, uncond_mask], dim=0)

        z = torch.randn(1, 4, latent_size, latent_size, device=device)
        fixed_z_cfg = torch.cat([z, z], dim=0)
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_DEFAULT_REGION"),
            bucket_name=os.environ.get("S3_BUCKET_NAME"),
            s3_prefix=os.environ.get("S3_PREFIX"),
            max_to_keep=5
        )

        del embedder
        torch.cuda.empty_cache()
    else:
        logger = create_logger(None)
        vae, fixed_embeddings, fixed_attention_mask, fixed_z_cfg = (
            None, None, None, None
        )
        checkpoint_dir = None
        checkpoint_manager = None


    
    model = DiffusionTransformer_models[args.model.name](input_size=latent_size)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    model = DDP(model.to(device), device_ids=[local_rank])
    diffusion = create_diffusion(timestep_respacing="1000")

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.training.optimizer.lr,
                            weight_decay=args.training.optimizer.weight_decay)

    start_epoch = 0
    train_steps = 0
    start_epoch_step = 0

    if args.logging.resume_ckpt:
        if rank == 0:
            logger.info(f"Resuming training from checkpoint: {args.logging.resume_ckpt}")

        checkpoint = torch.load(args.logging.resume_ckpt, map_location="cpu", weights_only=False)
        model.module.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        start_epoch = checkpoint.get("epoch", 0)
        train_steps = checkpoint.get("train_steps", 0)
        start_epoch_step = checkpoint.get("epoch_step", -1) + 1

        if rank == 0:
            logger.info("Successfully loaded checkpoint!")
            logger.info(
                f"=> Resuming at Epoch: {start_epoch}, Global Step: {train_steps}, Epoch Step: {start_epoch_step}")

        del checkpoint
        torch.cuda.empty_cache()


    dataset = Flickr8kDataset(
        root=args.dataset.data_path,
        latents_dir=args.dataset.latents_path,
        text_embs_dir=args.dataset.text_embeds_path,
        p_uncond=args.dataset.p_uncond
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.training.global_seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        sampler=sampler,
        num_workers=args.dataset.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.dataset.num_workers > 0),
    )

    logger.info(f"Dataset contains {len(dataset):,} images ({args.dataset.data_path})")

    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.training.epochs} epochs...")

    for epoch in range(start_epoch, args.training.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        for step, (x, y, mask) in enumerate(loader):
            if epoch == start_epoch and step < start_epoch_step:
                if step == 0 and rank == 0:
                    logger.info(
                        f"Skipping first {start_epoch_step} batches in epoch {epoch} to align with checkpoint...")
                continue

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y.unsqueeze(1), mask=mask)

            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            opt.zero_grad(set_to_none=True)

            loss.backward()
            opt.step()

            update_ema(ema, model.module)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.logging.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size

                logger.info(
                    f"(epoch={epoch:04d}, step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.logging.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "epoch": epoch,
                        "train_steps": train_steps,
                        "epoch_step": step,
                    }

                    checkpoint_path = f"{checkpoint_dir}/checkpoint_{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)

                    latest_path = f"{checkpoint_dir}/checkpoint_latest.pt"
                    torch.save(checkpoint, latest_path)

                    checkpoint_manager.manage_local()
                    if args.logging.is_save_to_s3:
                        checkpoint_manager.manage_s3(specific_checkpoint_path=checkpoint_path)

                    logger.info(f"Saved checkpoint to {checkpoint_path} and updated latest.pt")

                    if fixed_embeddings is not None:
                        logger.info(f"Generating 1 sample image for step {train_steps} with CFG...")
                        sample_model = (
                            ema if args.sampling.model_type == "ema"
                            else model.module
                        )
                        sample_model.eval()

                        vae.to(device)

                        with torch.inference_mode():
                            sample_model_kwargs = dict(
                                y=fixed_embeddings,
                                mask=fixed_attention_mask,
                                cfg_scale=args.sampling.cfg_scale,
                            )
                            samples = diffusion.p_sample_loop(
                                sample_model.forward_with_cfg,
                                fixed_z_cfg.shape,
                                fixed_z_cfg,
                                clip_denoised=False,
                                model_kwargs=sample_model_kwargs,
                                progress=False,
                                device=device,
                            )
                            samples, _ = samples.chunk(2, dim=0)
                            samples = vae.decode(latent=samples)
                            samples = (samples / 2 + 0.5).clamp(0, 1)

                            save_path = f"{experiment_dir}/sample.png"
                            torchvision.utils.save_image(samples, save_path)
                            logger.info(f"Saved generated sample to {save_path}")

                        vae.to("cpu")

                dist.barrier()

        dist.barrier()
        model.train()

    model.eval()
    logger.info("Training completed successfully!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    cmd_args = parser.parse_args()
    args = OmegaConf.load(cmd_args.config)
    main(args)
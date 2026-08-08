import numpy as np
import torch
import enum
from ai_engine.diffusion.diffusion_utils import discretized_gaussian_log_likelihood, normal_kl, mean_flat, extract_into_tensor


class ModelMeanType(enum.Enum):
    EPSILON = enum.auto()


class ModelVarType(enum.Enum):
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start, beta_end = scale * 0.0001, scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


class GaussianDiffusion:
    def __init__(self, *, betas, model_mean_type, model_var_type, loss_type):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.betas = np.array(betas, dtype=np.float64)
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])
        self.posterior_mean_coef1 = self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, clip_denoised=True, model_kwargs=None):
        if model_kwargs is None: model_kwargs = {}
        B, C = x.shape[:2]
        model_output = model(x, t, **model_kwargs)

        if self.model_var_type == ModelVarType.LEARNED_RANGE:
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            min_log = extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = extract_into_tensor(np.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = torch.exp(model_log_variance)
        else:
            model_variance = extract_into_tensor(self.posterior_variance, t, x.shape)
            model_log_variance = extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)

        pred_xstart = (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * model_output
        )
        if clip_denoised: pred_xstart = pred_xstart.clamp(-1, 1)

        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        return {"mean": model_mean, "variance": model_variance, "log_variance": model_log_variance,
                "pred_xstart": pred_xstart}

    def p_sample(self, model, x, t, clip_denoised=True, model_kwargs=None):
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        noise = torch.randn_like(x)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1))))
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(self,
                      model,
                      shape,
                      noise=None,
                      clip_denoised=True,
                      model_kwargs=None,
                      device=None,
                      progress=False):
        if device is None: device = next(model.parameters()).device
        img = noise if noise is not None else torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(model, img, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
                img = out["sample"]
        return img

    def _vb_terms_bpd(self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        out = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        kl = mean_flat(normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])) / np.log(2.0)
        decoder_nll = -discretized_gaussian_log_likelihood(x_start, means=out["mean"],
                                                           log_scales=0.5 * out["log_variance"])
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)
        return {"output": torch.where((t == 0), decoder_nll, kl), "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        if model_kwargs is None: model_kwargs = {}
        if noise is None: noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        terms = {}

        model_output = model(x_t, t, **model_kwargs)
        if self.model_var_type == ModelVarType.LEARNED_RANGE:
            B, C = x_t.shape[:2]
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
            terms["vb"] = self._vb_terms_bpd(
                model=lambda *args, r=frozen_out: r, x_start=x_start, x_t=x_t, t=t, clip_denoised=False
            )["output"]
            if self.loss_type == LossType.RESCALED_MSE:
                terms["vb"] *= self.num_timesteps / 1000.0

        terms["mse"] = mean_flat((noise - model_output) ** 2)
        terms["loss"] = terms["mse"] + terms.get("vb", 0.0)
        return terms
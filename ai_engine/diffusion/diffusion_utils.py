import numpy as np
import torch

def normal_kl(mean1, logvar1, mean2, logvar2):
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    logvar1, logvar2 = [x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor) for x in (logvar1, logvar2)]
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))

def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    left_edge = x - 1.0 / 255.0
    right_edge = x + 1.0 / 255.0
    inv_stdv = torch.exp(-log_scales)

    plus_in = (right_edge - means) * inv_stdv
    cdf_plus = approx_standard_normal_cdf(plus_in)

    min_in = (left_edge - means) * inv_stdv
    cdf_min = approx_standard_normal_cdf(min_in)

    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
        x < -0.999,
        torch.log(cdf_plus.clamp(min=1e-12)),
        torch.where(
            x > 0.999,
            torch.log((1.0 - cdf_min).clamp(min=1e-12)),
            torch.log(cdf_delta.clamp(min=1e-12)),
        ),
    )

    return log_probs

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)
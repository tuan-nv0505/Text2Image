from ai_engine.diffusion.gaussian_diffusion import ModelMeanType, ModelVarType, LossType, get_named_beta_schedule
from ai_engine.diffusion.respace import space_timesteps, SpacedDiffusion


def create_diffusion(
        timestep_respacing,
        noise_schedule="linear",
        learn_sigma=True,
        rescale_learned_sigmas=False,
        diffusion_steps=1000
):
    betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
    loss_type = LossType.RESCALED_MSE if rescale_learned_sigmas else LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.LEARNED_RANGE if learn_sigma else ModelVarType.FIXED_LARGE,
        loss_type=loss_type
    )
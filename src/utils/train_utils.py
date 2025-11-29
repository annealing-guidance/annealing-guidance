import os
import torch
from torch.utils.data import DataLoader
import torch
from src.data.dataset import LaionDataset

def get_data_loader(config):
    """
    Get DataLoaders for training and validation splits of the generated dataset.

    Args:
        config (dict): Configuration dictionary containing dataset parameters.
        device (torch.device, optional): Device to move data to (e.g., CPU or GPU).

    Returns:
        tuple: A tuple containing (train_loader, val_loader).
    """
    # Dataset parameters
    batch_size = config["training"]["batch_size"]
    image_root = config["training"]["image_root"]
    dataset = LaionDataset(image_root)

    # Create DataLoader for training
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.manual_seed(config['seed']),
        pin_memory=True
    )

    return dataloader

def save_model(config, guidance_scale_model, step, timestamp):
    dict_to_save = {
        'config': config,
        'guidance_scale_model': guidance_scale_model.state_dict(),
    }

    out_dir = config['training']['out_dir']
    checkpoint_dir = f'{out_dir}/checkpoints_{timestamp}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(dict_to_save, f'{checkpoint_dir}/checkpoint_step_{step}.pt')

def get_timestep(pipeline, batch_size=1):
    n_timesteps = len(pipeline.scheduler.timesteps)
    timesteps_indices = torch.randint(1, n_timesteps, size=[batch_size])
    timesteps = pipeline.scheduler.timesteps[n_timesteps- 1 - timesteps_indices]
    timesteps = timesteps.to(device=pipeline.device)
    return timesteps


def calc_loss(noise_pred, noise_gt, delta_t_minus_one, l):
    loss = torch.tensor(0.0, device=noise_pred.device)
    diff_loss = torch.tensor(0.0, device=noise_pred.device)
    eps_loss = torch.tensor(0.0, device=noise_pred.device)

    # epsilon loss
    squared_errors = ((noise_pred - noise_gt) ** 2).mean(dim=[1,2,3])  # mean over features per sample
    eps_loss = ((1-l) * squared_errors).mean()
    loss+=eps_loss

    # delta loss
    squared_errors = (delta_t_minus_one ** 2).mean(dim=[1,2,3])
    diff_loss = (l * squared_errors).mean()
    loss += diff_loss


    return loss


def encode_prompt(
    pipeline, 
    prompt,
    original_size=(1024, 1024),
    crops_coords_top_left=(0, 0),
    target_size=(1024, 1024),
):
    device = pipeline.device
    dtype = pipeline.unet.dtype

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1)

    add_time_ids = pipeline._get_add_time_ids(
        original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype)

    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    add_time_ids = torch.cat((add_time_ids, add_time_ids), dim=0)


    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)
    add_time_ids = add_time_ids.to(device=device, dtype=dtype)
    add_time_ids = add_time_ids.repeat(len(prompt), 1)

    added_cond_kwargs = {'text_embeds': pooled_prompt_embeds, 'time_ids': add_time_ids}
    return prompt_embeds, added_cond_kwargs


def denoise_single_step(
    pipeline,
    latents,
    prompt_embeds,
    timestep,
    added_cond_kwargs,
):
    timesteps = torch.cat([timestep] * 2) # duplicate for uncond pred
    latent_model_input = torch.cat([latents] * 2) # duplicate for uncond pred
    latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, timesteps)
    return pipeline.unet(
        latent_model_input,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )

def to_noisy_latents(pipeline, image, timestep, size=(1024, 1024)):
    # x_0 -> z_0
    with torch.no_grad():
        vae = pipeline.vae.to(torch.float32)  # run vae in float32 always to avoid black images
        image = image.to(device=vae.device, dtype=vae.dtype)
        image = torch.nn.functional.interpolate(image, size=size, mode='bilinear')
        latents = vae.encode(image).latent_dist.sample(generator=None) 
        latents = latents * vae.config.scaling_factor # z_0
        
    latents = latents.to(dtype=pipeline.unet.dtype)

    noise = torch.randn_like(latents)
    
    noisy_latents = pipeline.scheduler.add_noise(latents, noise, timestep) # z_0 -> z_t

    return noisy_latents, noise

def linear_schedule(t: torch.Tensor, tau1: float, tau2: float) -> torch.Tensor:
    """
    CADS annealing schedule function that returns values in [0, 1].
    Works with scalar or tensor inputs for t.
    """
    # Compute gamma for all t
    gamma = (tau2 - t) / (tau2 - tau1)

    # Apply piecewise clamp: 1 before tau1, 0 after tau2, linear in between
    gamma = torch.clamp(gamma, min=0.0, max=1.0)

    return gamma

def add_noise_to_prompt(y, gamma, noise_scale, psi, rescale=False):
    """ CADS adding noise to the condition

    Arguments:
    y: Input conditioning
    gamma: Noise level w.r.t t
    noise_scale (float): Noise scale
    psi (float): Rescaling factor
    rescale (bool): Rescale the condition
    """
    eps = 1e-6 
    noise = torch.randn_like(y)
    gamma = gamma.view(-1, *[1]*(y.ndim-1))
    y_noised = torch.sqrt(gamma) * y + noise_scale * torch.sqrt(1 - gamma) * noise

    if not rescale:
        return y_noised

    # per-sample mean/std (over all but batch dim if present)
    dims = tuple(range(1, y.ndim)) if y.ndim > 1 else ()
    y_mean, y_std = y.mean(dims, keepdim=True), y.std(dims, keepdim=True) + eps
    yn_mean, yn_std = y_noised.mean(dims, keepdim=True), y_noised.std(dims, keepdim=True) + eps

    y_scaled = (y_noised - yn_mean) / yn_std * y_std + y_mean
    return psi * y_scaled + (1 - psi) * y_noised
    
def prompt_add_noise(
    prompt_embeds,
    added_cond_kwargs,
    timestep,
    n_timesteps,
    add_noise,
    noise_scale,
    rescale,
    psi,
    t1,
    t2
):

    if add_noise:
        t = timestep/n_timesteps
        gamma = linear_schedule(t, t1, t2)

        # split to negative prompt embed and prompt embed
        negative_prompt_embeds, cond_prompt_embeds = prompt_embeds.chunk(2)
        cond_prompt_embeds = add_noise_to_prompt(cond_prompt_embeds, gamma, noise_scale, psi, rescale=rescale)
        prompt_embeds = torch.cat([negative_prompt_embeds, cond_prompt_embeds], dim=0)

        negative_pooled_prompt_embeds, pooled_cond_prompt_embeds = added_cond_kwargs['text_embeds'].chunk(2)
        pooled_cond_prompt_embeds = add_noise_to_prompt(pooled_cond_prompt_embeds, gamma, noise_scale, psi, rescale=rescale)
        added_cond_kwargs['text_embeds'] = torch.cat([negative_pooled_prompt_embeds, pooled_cond_prompt_embeds], dim=0)

    return prompt_embeds, added_cond_kwargs

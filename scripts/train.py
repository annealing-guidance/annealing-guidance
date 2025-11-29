import math
import torch
import datetime
import tqdm
import src.utils.model_utils as model_utils
import src.utils.train_utils as train_utils


def train(config, pipeline, model, optimizer, dataloader):
    train_config = config['training']
    max_steps = train_config['max_steps']
    max_epochs = math.ceil(max_steps / len(dataloader))
    accumulation_steps = max(train_config.get('accumulation_steps', 1), 1)
    
    train_end = False
    global_step = 0
    
    datetime_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(max_epochs):
        for batch in tqdm.tqdm(dataloader):
            model.train()
            prompts, images = batch

            images = images.to(pipeline.device)
            
            loss = forward_pass(config, pipeline, model, images, prompts)


            loss = loss / accumulation_steps  # Normalize loss by accumulation steps
            loss.backward()

            if (global_step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if global_step > 0 and global_step % config['training']['save_interval'] == 0:
                print(f"Saving model at step {global_step}...")
                train_utils.save_model(config, model, global_step, datetime_timestamp)


            global_step += 1
            if global_step > max_steps:
                train_end = True
                break

        if train_end:
            break


def forward_pass(
    config,
    pipeline,
    model,
    images,
    prompts,
):
    batch_size = images.size(0)
    
    # Select lambda values
    l = torch.rand(batch_size).to(pipeline.unet.device)

    # Select timestep values
    timestep = train_utils.get_timestep(pipeline, batch_size=batch_size)

    # Get noisy latents and ground truth noise
    # x_0 -> z_t
    noisy_latents, noise_gt = train_utils.to_noisy_latents(pipeline, images, timestep) # (z_t, eps)

    # Get prompt embeddings
    with torch.no_grad():
        prompt_embeds, added_cond_kwargs = train_utils.encode_prompt(pipeline, prompts)

    # Use CADS to add noise to conditioning signal (if enabled)
    prompt_embeds, added_cond_kwargs = train_utils.prompt_add_noise(
        prompt_embeds,
        added_cond_kwargs,
        timestep,
        pipeline.scheduler.config['num_train_timesteps'],
        **config['training']['prompt_noise']
    )

    # Predict epsilon_null + epsilon_cond
    noise_pred = train_utils.denoise_single_step(
        pipeline,
        noisy_latents,
        prompt_embeds,
        timestep,
        added_cond_kwargs,
    )[0]
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    
    # Predict guidance scale
    # noise_pred, guidance_scale, _ = pipeline.perform_guidance(noise_pred_uncond, noise_pred_text, model.cfg, timestep, model, l=l)
    guidance_scale_pred = model(timestep, l, noise_pred_uncond, noise_pred_text)
    
    # Apply classifier free guidance
    noise_pred = noise_pred_uncond + guidance_scale_pred * (noise_pred_text - noise_pred_uncond)
    
    # Renoise to next latent
    # z_t -> z_{t+1}
    results =  pipeline.scheduler.step(noise_pred, timestep, noisy_latents, return_dict=True, noise_pred_uncond=noise_pred_uncond)
    pred_latents_prev = results['prev_sample']


    # Calculate delta_t_minus_one
    # Reverse scheduler_timesteps for ascending order in search
    scheduler_timesteps = pipeline.scheduler.timesteps.clone().to(device=timestep.device)
    scheduler_timesteps_reversed = scheduler_timesteps.flip(0)

    # Find indices in the reversed array
    timestep_indices = torch.searchsorted(scheduler_timesteps_reversed, timestep, right=True) - 1
    valid_indices = (timestep_indices - 1) >= 0
    if valid_indices.any():
        timestep_indices = timestep_indices[valid_indices]
        timestep_prev = scheduler_timesteps_reversed[timestep_indices - 1]
        pred_latents_prev = pred_latents_prev[valid_indices]
        prompt = [p for p, valid in zip(prompts, valid_indices.cpu().numpy()) if valid]
        # Filter other necessary tensors like prompt_embeds or added_cond_kwargs if they have a batch dimension

        with torch.no_grad():
            prompt_embeds, added_cond_kwargs = train_utils.encode_prompt(pipeline, prompt)
            
        noise_pred_prev = train_utils.denoise_single_step(
            pipeline,
            pred_latents_prev,
            prompt_embeds,  # Assuming prompt_embeds has a batch dimension
            timestep_prev,
            {k: v[valid_indices] if isinstance(v, torch.Tensor) and v.shape[0] == valid_indices.shape[0] else v
            for k, v in added_cond_kwargs.items()},
        )[0]

        noise_pred_uncond_prev, noise_pred_text_prev = noise_pred_prev.chunk(2, dim=0)
        delta_t_minus_one = noise_pred_uncond_prev - noise_pred_text_prev
        
    # calc loss
    loss = train_utils.calc_loss(noise_pred, noise_gt, delta_t_minus_one, l)

    return loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_path = 'scripts/config.yaml'
config, pipeline, guidance_scale_network = model_utils.load_models(
    config_path=config_path,
    device=device,
)

optimizer = torch.optim.AdamW(guidance_scale_network.parameters(), **config['training']['optimizer_kwargs'])
dataloader = train_utils.get_data_loader(config)


train(config, pipeline, guidance_scale_network, optimizer, dataloader)

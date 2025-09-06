import torch

from src.schedulers.my_scheduling_ddim import MyDDIMScheduler
from omegaconf import OmegaConf
from src.pipelines.my_pipeline_stable_diffusion_xl import MyStableDiffusionXLPipeline
from src.model.guidance_scale_model import ScalarMLP

def load_guidance_scale_model(config, scheduler, state_dict=None, device=None, dtype=None):
    device = torch.device('cuda') if device is None else device
    model = ScalarMLP(
        scheduler = scheduler,
        **config['guidance_scale_model'],
    )
    if state_dict is not None:
        model.load_state_dict(state_dict['guidance_scale_model'], strict=False)

    model.to(device, dtype=dtype)
    model.device = device
    model.dtype = dtype
    return model

def load_models(checkpoint_path=None, config_path=None, device=None):
    # Assert either checkpoint_path or config_path is provided, but not both
    assert checkpoint_path is not None or config_path is not None, "Either checkpoint_path or config_path must be provided"
    assert not (checkpoint_path is not None and config_path is not None), "Both checkpoint_path and config_path cannot be provided"

    state_dict, config = load_config(checkpoint_path=checkpoint_path, config_path=config_path)
    dtype = get_dtype(config)
    pipeline = load_pipeline(config, device=device, dtype=dtype)
    guidance_scale_network = load_guidance_scale_model(config, pipeline.scheduler, state_dict=state_dict, dtype=dtype)
    return config, pipeline, guidance_scale_network

def get_dtype(config):
    dtype = torch.float16 if config.get('low_memory', True) else torch.float32
    return dtype

def load_config(config_path=None, checkpoint_path=None):
    if checkpoint_path is None:
        state_dict = None
        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)
    else:
        state_dict = torch.load(checkpoint_path, weights_only=True)
        config = state_dict['config']

    return state_dict, config

def load_scheduler(config):
    num_timesteps = config['diffusion']['num_timesteps']
    scheduler_type = config['diffusion'].get('scheduler_type', 'ddim')
    scheduler = MyDDIMScheduler(**config['diffusion']['scheduler_kwargs'])
    scheduler.set_timesteps(num_timesteps)

    return scheduler
def load_pipeline(config, device=None, dtype=None):
    
    device = torch.device('cuda') if device is None else device
    dtype = torch.float16 if dtype is None else dtype

    scheduler = load_scheduler(config)

    model_type = config['diffusion'].get('model_type', 'sdxl').lower()
    pipeline = MyStableDiffusionXLPipeline.from_pretrained(
        config['diffusion']['model_id'], scheduler=scheduler, torch_dtype=dtype)

    pipeline.to(device)
    

    # pipeline.enable_xformers_memory_efficient_attention()
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)

    return pipeline

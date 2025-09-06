import os
import torch
import src.utils.model_utils as model_utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# =====================
#      CFG SECTION
# =====================

cfg_scale = 10  # Classifier-Free Guidance scale

# =====================
#      OURS SECTION
# =====================

model_path = 'src/model/checkpoints/checkpoint.pt'  # Annealing, with prompt renoise, lambda=0.4
guidance_lambda = 0.4

config, pipeline, guidance_scale_network = model_utils.load_models(
    checkpoint_path=model_path,
    device=device,
)

# =====================
#      SAMPLING
# =====================
seed = 32426
prompt = 'A mid-air dog practicing karate in a Japanese dojo, wearing a white gi with a black belt, mid-pose on wooden floors with soft lighting and a focused expression.'



out_dir = 'samples'
os.makedirs(out_dir, exist_ok=True)


# CFG (Vanilla)
# Disable null-direction renoising
pipeline.scheduler.config.pred_sample_direction_with_null = False
img_base = pipeline(
    prompt=prompt,
    guidance_scale=cfg_scale,
    generator=torch.Generator(pipeline.unet.device).manual_seed(seed)
).images[0]
img_base.save(f'{out_dir}/vanilla.jpg')

# Ours
# Enable null-direction renoising
pipeline.scheduler.config.pred_sample_direction_with_null = True
img_ours = pipeline(
    prompt=prompt,
    guidance_scale_model=guidance_scale_network,
    guidance_lambda=guidance_lambda,
    generator=torch.Generator(pipeline.unet.device).manual_seed(seed)
).images[0]
img_ours.save(f'{out_dir}/ours.jpg')

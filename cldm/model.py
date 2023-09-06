import os
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def parse_model_config(config_path):
    config = OmegaConf.load(config_path)
    model_config = config.model
    return model_config


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model


def check_inference_shape(model, batch_size: int = 4):
    H, W = 704, 512
    # H, W = 1024, 768
    device = torch.device("cuda")
    eps = model.to(device).apply_model(
        x_noisy={
            "noisy": torch.randn(batch_size, 4, H // 8, W // 8).to(device),  # noise
            "z_ag": torch.randn(batch_size, 4, H // 8, W // 8).to(device),  # cloth-agnostic person
        },
        t=torch.full((batch_size,), 1, dtype=torch.long).to(device),  # timestamp
        cond={
            "c_concat": [torch.randn(batch_size, 3, H, W).to(device)],  # openpose
            "c_crossattn": [
                model.get_learned_conditioning(
                    {"text": ["hello world"] * batch_size,
                     "image": torch.randn(batch_size, 3, 224, 224).to(device)}
                )
            ]  # prompt
        }
    )
    print("[check] inference successes! Shape is: ", eps.shape)

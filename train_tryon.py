from share import *

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from ldm.data.tryon_dataset import TryOnDataset


def check_inference_shape(model):
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
    print("[main] inference successes! Shape is: ", eps.shape)


# Configs
resume_path = './models/control_sd15_openpose.pth'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/tryon_v15.yaml').cpu()
empty_state_dict = model.state_dict()

pretrained_cldm_weights = load_state_dict(resume_path, location='cpu')
pretrained_cldm_weights["model.diffusion_model.input_blocks.0.0.weight"] = \
    pretrained_cldm_weights["model.diffusion_model.input_blocks.0.0.weight"].repeat(1, 2, 1, 1)  # doubled in_channels
pretrained_cldm_weights["control_model.input_blocks.0.0.weight"] = \
    pretrained_cldm_weights["control_model.input_blocks.0.0.weight"].repeat(1, 2, 1, 1)  # doubled in_channels

pretrained_dict = {k: v for k, v in pretrained_cldm_weights.items() if k in empty_state_dict}
empty_state_dict.update(pretrained_dict)
model.load_state_dict(empty_state_dict)
print("[main] model loaded.")

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = TryOnDataset(
    root="/cfs/yuange/datasets/VTON-HD",
    mode="train",
)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(accelerator="gpu", devices="0,", precision=32, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)

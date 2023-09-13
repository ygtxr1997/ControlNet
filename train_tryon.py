from share import *

import os, datetime, argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict, parse_model_config
from cldm.dataloader import create_dataset, parse_training_config, get_parser


cmd_parser = get_parser()
cmd_opt, _ = cmd_parser.parse_known_args()
log_dir = cmd_opt.log_dir
batch_size = cmd_opt.batch_size

""" Check resuming """
resume_project_folder = cmd_opt.resume_project_folder
resume_ckpt_name = cmd_opt.resume_ckpt_name
resume_cfg = f"./logs/{resume_project_folder}/tryon_v15.yaml"
resume_ckpt_path = f"./logs/{resume_project_folder}/version_0/checkpoints/{resume_ckpt_name}"
resume_zero_input_blocks: bool = cmd_opt.resume_zero_input_blocks
if os.path.exists(resume_cfg) and os.path.exists(resume_ckpt_path):
    resume_training = True
    cfg_file = resume_cfg
else:
    resume_training = False
    cfg_file = cmd_opt.cfg_file
print(f"[main] config file loaded from: {cfg_file}. (Resume checkpoint?: {resume_training})")


""" Configs """
model_config = parse_model_config(cfg_file)
train_config = parse_training_config(cfg_file)
sd_path = train_config.sd_path
batch_size = train_config.batch_size if batch_size <=0 else batch_size
logger_freq = train_config.logger_freq
learning_rate = train_config.learning_rate
sd_locked = train_config.sd_locked
only_mid_control = train_config.only_mid_control

zero_init_input_blocks = train_config.zero_init_input_blocks

now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
project_name = now
project_folder = os.path.join(log_dir, project_name)


""" First use cpu to load models. Pytorch Lightning will automatically move it to GPUs. """
model: pl.LightningModule = create_model(cfg_file).cpu()
print(f"[main] model created according: {cfg_file}.")

empty_state_dict = model.state_dict()
if not resume_training:
    pretrained_cldm_weights = load_state_dict(sd_path, location='cpu')
    channel04_unet_weight = pretrained_cldm_weights["model.diffusion_model.input_blocks.0.0.weight"]  # (320,4,3,3)
    channel04_control_weight = pretrained_cldm_weights["control_model.input_blocks.0.0.weight"]
    channel03_hint_weight = pretrained_cldm_weights["control_model.input_hint_block.0.weight"]
    if not zero_init_input_blocks:
        channel48_unet_weight = channel04_unet_weight
        channel48_control_weight = channel04_control_weight
        channel89_unet_weight = channel04_unet_weight[:, 0].unsqueeze(1)
        channel89_control_weight = channel04_control_weight[:, 0].unsqueeze(1)
        channel36_hint_weight = channel03_hint_weight
    else:
        channel48_unet_weight = torch.zeros_like(channel04_unet_weight)
        channel48_control_weight = torch.zeros_like(channel04_control_weight)
        channel89_unet_weight = torch.zeros_like(channel04_unet_weight[:, 0].unsqueeze(1))
        channel89_control_weight = torch.zeros_like(channel04_control_weight[:, 0].unsqueeze(1))
        channel36_hint_weight = torch.zeros_like(channel03_hint_weight)

    pretrained_cldm_weights["model.diffusion_model.input_blocks.0.0.weight"] = \
        torch.cat([channel04_unet_weight,
                   channel48_unet_weight,
                   channel89_unet_weight], dim=1)
    pretrained_cldm_weights["control_model.input_blocks.0.0.weight"] = \
        torch.cat([channel04_control_weight,
                   channel48_control_weight,
                   channel89_control_weight], dim=1)
    if model_config.params.feed_cloth_to_controlnet:
        pretrained_cldm_weights["control_model.input_hint_block.0.weight"] = \
            torch.cat([channel03_hint_weight,
                       channel36_hint_weight], dim=1)

    pretrained_dict = {k: v for k, v in pretrained_cldm_weights.items() if k in empty_state_dict}
    empty_state_dict.update(pretrained_dict)
    model.load_state_dict(empty_state_dict)
    print(f"[main] model loaded from: {sd_path}.")
else:
    resume_ckpt_path = f"./logs/{resume_project_folder}/version_0/checkpoints/{resume_ckpt_name}"
    resume_weight = torch.load(resume_ckpt_path, map_location="cpu")["state_dict"]

    if resume_zero_input_blocks:
        resume_weight["model.diffusion_model.input_blocks.0.0.weight"][:, 4:] = 0.  # (320,4+4+1,3,3)
        resume_weight["control_model.input_blocks.0.0.weight"][:, 4:] = 0.
        if model_config.params.feed_cloth_to_controlnet:
            resume_weight["control_model.input_hint_block.0.weight"][:, 3:] = 0.

    pretrained_dict = {k: v for k, v in resume_weight.items() if k in empty_state_dict}
    empty_state_dict.update(pretrained_dict)
    model.load_state_dict(empty_state_dict)
    print(f"[main] model resume successfully, from: {resume_ckpt_path}. "
          f"ZeroInput={resume_zero_input_blocks}. ")

model.learning_rate = train_config.learning_rate
model.sd_locked = train_config.sd_locked
model.only_mid_control = train_config.only_mid_control

""" Misc """
dataset = create_dataset(train_config.dataset_train_config)
dataloader = DataLoader(dataset, num_workers=2, batch_size=batch_size, shuffle=True)
logger = ImageLogger(
    batch_frequency=logger_freq,
    project_name=project_name,
    source_cfg_path=cfg_file,
)
tensorboard_logger = TensorBoardLogger(
    save_dir=log_dir,
    name=now,
)
trainer = pl.Trainer(strategy=DDPStrategy(find_unused_parameters=True), accelerator="gpu",
                     devices=cmd_opt.gpus, precision=32,
                     callbacks=[logger],
                     logger=tensorboard_logger)


""" Train! """
trainer.fit(model, dataloader)

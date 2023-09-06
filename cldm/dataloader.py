import os, argparse
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="The root folder of all logs.",
    )
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="./models/tryon_v15.yaml",
        help="The model and training config file.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help="The indices of GPUs.",
    )
    parser.add_argument(
        "--resume_project_folder",
        type=str,
        default="",
        help="Resume training from which sub-folder?",
    )
    parser.add_argument(
        "--resume_ckpt_name",
        type=str,
        default="",
        help="Resume training from which checkpoint under the sub-folder?",
    )
    parser.add_argument(
        "--resume_zero_input_blocks",
        type=str2bool,
        default=False,
        help="If true, resume weights but set input blocks to zero.",
    )

    return parser


def parse_training_config(config_path):
    config = OmegaConf.load(config_path)
    config_train = config.train
    return config_train


def create_dataset(dataset_config):
    dataset = instantiate_from_config(dataset_config)
    return dataset


if __name__ == "__main__":
    cfg_file = "./models/tryon_v15.yaml"
    cfg_train = parse_training_config(cfg_file)
    print(cfg_train)

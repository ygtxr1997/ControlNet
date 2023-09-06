import os.path

from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from cldm.model import create_model, load_state_dict, parse_model_config
from cldm.ddim_hacked import DDIMSamplerTryOn
from ldm.data.tryon_dataset import TryOnDataset


apply_openpose = OpenposeDetector()

project_folder = "2023-09-04T15-46-02"
ckpt_name = "epoch=131-step=48048.ckpt"
config_file = f"./logs/{project_folder}/tryon_v15.yaml"
model = create_model(config_file).cpu()
model_config = parse_model_config(config_file)
# model.load_state_dict(load_state_dict('./models/control_sd15_openpose.pth', location='cuda'))
weights_folder = f"./logs/{project_folder}/version_0/checkpoints/"
weights_path = os.path.join(weights_folder, ckpt_name)
if not os.path.exists(weights_path):
    weights_path = os.path.join(weights_folder, os.listdir(weights_folder)[-1])
    print(f"[main][Warning] {ckpt_name} doesn't exist, load the last file {weights_path}.")
weights = torch.load(weights_path, map_location="cpu")["state_dict"]
empty_weights = model.state_dict()
load_weights = {}
for k, v in weights.items():
    if k in empty_weights:
        load_weights[k] = v
empty_weights.update(load_weights)
model.load_state_dict(empty_weights)
print("[main] state_dict loaded successfully! From:", os.path.join("logs", project_folder))
model = model.cuda()

dataset = TryOnDataset(
    root="/cfs/yuange/datasets/VTON-HD",
    mode="test",
    use_warp_pasted_agnostic=True,
)
ddim_sampler = DDIMSamplerTryOn(
    model,
    tryon_dataset=dataset
)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, test_index):
    with torch.no_grad():
        index = test_index
        item = dataset[index]

        person = item["jpg"]
        cloth_masked = item["txt"]["cloth"]
        agnostic = item["agnostic"]
        openpose = item["hint"]

        input_image = HWC3(input_image)
        detected_map, _ = apply_openpose(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)  # (H,W,3), in [0,255]

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()  # (B,3,H,W), in [0,1]

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        device = torch.device("cuda")
        prompt = "an image"
        agnostic, openpose, cloth_masked, agnostic_mask = ddim_sampler.get_apply_input(
            index, num_samples=num_samples, device=device)
        control_list = [openpose]
        if model_config.params.feed_cloth_to_controlnet:
            cloth_control = F.interpolate(cloth_masked, size=openpose.shape[-2:], mode="bilinear", align_corners=True)
            control_list.append(cloth_control)

        cond_in_dict = {
            "text": [prompt + ', ' + a_prompt] * num_samples,
            "image": cloth_masked,
        }
        un_cond_in_dict = {
            "text": [n_prompt] * num_samples,
            "image": torch.zeros_like(cloth_masked).to(cloth_masked.device),
        }
        apply_input = {"z_x": None, "z_ag": agnostic, "z_ag_m": agnostic_mask}
        # apply_input = {"z_x": None, "z_ag": torch.randn_like(agnostic).to(agnostic.device)}
        cond = {"c_concat": control_list, "c_crossattn": [model.get_learned_conditioning(cond_in_dict)]}
        un_cond = {"c_concat": None if guess_mode else control_list,
                   "c_crossattn": [model.get_learned_conditioning(un_cond_in_dict)]}
        shape = (4, H // 8, W // 8)
        shape = (agnostic.shape[1], agnostic.shape[2], agnostic.shape[3])

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond,
                                                     x_T=apply_input)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)  # in [-1,1]
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Human Pose")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                test_index = gr.Slider(label="Test Index", minimum=0, maximum=2031, value=0, step=1)
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                detect_resolution = gr.Slider(label="OpenPose Resolution", minimum=128, maximum=1024, value=512, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, test_index]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0')

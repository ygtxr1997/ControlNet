import os
import time

import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
from einops import rearrange

from annotator.clothwarp.pfafn_test.models.afwm import AFWM
from annotator.clothwarp.pfafn_test.models.networks import load_checkpoint
from annotator.clothwarp.pfafn_test.options.test_options import TestOptions


class PFAFNImageInfer(object):
    def __init__(self, device,
                 ckpt_path: str = "./models/warp_viton.pth"):
        self.opt = TestOptions().parse(verbose=False)
        self.device = device

        self.opt.warp_checkpoint = ckpt_path
        self.opt.label_nc = 13
        self.fine_height = self.opt.fineSize  # height:512
        self.fine_width =  int(self.fine_height / 1024 * 768)

        opt = self.opt
        warp_model = AFWM(opt, 3 + opt.label_nc)
        load_checkpoint(warp_model, opt.warp_checkpoint)
        warp_model = warp_model.eval()
        warp_model.to(device)
        self.warp_model = warp_model

        print(f"[PFAFNImageInfer] model loaded from {opt.warp_checkpoint}.")

    @torch.no_grad()
    def infer(self, parse_agnostic: torch.Tensor, dense_pose: torch.Tensor,
              cloth: torch.Tensor, cloth_mask: torch.Tensor,
              out_hw: tuple = None,
              ):
        """
        Like forward().
        @param parse_agnostic: (B,{0-19},768,1024), in {0,1}
        @param dense_pose: (B,RGB,768,1024), in [-1,1]
        @param cloth: (B,RGB,768,1024), in [-1,1]
        @param cloth_mask: (B,1,768,1024), in [0,1]
        @param out_hw: output H and W
        @returns: {"warped_cloth":(B,RGB,768,1024) in [-1,1], "warped_mask":(B,1,768,1024) in [0,1]}
        """
        ih, iw = parse_agnostic.shape[2:]
        th, tw = (256, 192)
        parse_agnostic_down = F.interpolate(parse_agnostic, size=(th, tw), mode='nearest')
        dense_posed_down = F.interpolate(dense_pose, size=(th, tw), mode='bilinear', align_corners=True)
        cloth_down = F.interpolate(cloth, size=(th, tw), mode='bilinear', align_corners=True)
        cloth_mask_down = F.interpolate(cloth_mask, size=(th, tw), mode='nearest')
        cloth_mask_down = torch.FloatTensor((cloth_mask_down.cpu().numpy() > 0.5).astype(float)).to(cloth.device)

        cond = torch.cat([parse_agnostic_down, dense_posed_down], dim=1)
        image = torch.cat([cloth_down, cloth_mask_down], dim=1)

        warped_cloth, last_flow = self.warp_model(cond, cloth_down)
        warped_mask = F.grid_sample(cloth_mask_down, last_flow.permute(0, 2, 3, 1),
                                    mode='bilinear', padding_mode='zeros')

        if ih != 256:
            last_flow = F.interpolate(last_flow, size=(ih, iw), mode='bilinear', align_corners=True)
            warped_cloth = F.grid_sample(cloth, last_flow.permute(0, 2, 3, 1),
                                         mode='bilinear', padding_mode='border')
            warped_mask = F.grid_sample(cloth_mask, last_flow.permute(0, 2, 3, 1),
                                        mode='bilinear', padding_mode='zeros')

        if out_hw is not None:
            warped_cloth = F.interpolate(warped_cloth, size=out_hw, mode="bilinear", align_corners=True)
            warped_mask = F.interpolate(warped_mask, size=out_hw, mode="bilinear", align_corners=True)
        return {
            "warped_cloth": warped_cloth.clamp(-1., 1.),
            "warped_mask": warped_mask.clamp(0., 1.),
        }


def seg_to_onehot(seg: np.ndarray, seg_nc: int = 13, device: torch.device = "cuda"):
    # parse map
    labels = {
        0: ['background', [0, 10]],
        1: ['hair', [1, 2]],
        2: ['face', [4, 13]],
        3: ['upper', [5, 6, 7]],
        4: ['bottom', [9, 12]],
        5: ['left_arm', [14]],
        6: ['right_arm', [15]],
        7: ['left_leg', [16]],
        8: ['right_leg', [17]],
        9: ['left_shoe', [18]],
        10: ['right_shoe', [19]],
        11: ['socks', [8]],
        12: ['noise', [3, 11]]
    }

    h, w = seg.shape
    x = torch.from_numpy(seg[None]).long()
    one_hot = torch.zeros((20, h, w), dtype=torch.float32)
    one_hot = one_hot.scatter_(0, x, 1.0)
    ret_one_hot = torch.zeros((seg_nc, h, w), dtype=torch.float32)
    for i in range(len(labels)):
        for label in labels[i][1]:
            ret_one_hot[i] += one_hot[label]
    return ret_one_hot.unsqueeze(0).to(device)


def hwc_to_nchw(x: np.ndarray, device: torch.device = "cuda", zero_center: bool = True, norm: bool = True):
    x = torch.FloatTensor(x.astype(np.uint8)).to(device=device)
    if norm:
        if zero_center:
            x = (x / 127.5 - 1.)
        else:
            x = x / 255.
    if x.ndim == 2:  # gray, only (h,w)
        x = x.unsqueeze(-1)
    x = x.unsqueeze(0)
    x = rearrange(x, "n h w c -> n c h w").contiguous()
    return x


def nchw_to_hwc(x: torch.Tensor, b_idx: int = 0, zero_center: bool = True):
    x = rearrange(x, "n c h w -> n h w c").contiguous()
    x = x[b_idx]
    if zero_center:
        x = (x + 1.) * 127.5
    else:
        x = x * 255.
    x = x.cpu().numpy().astype(np.uint8)
    return x


if __name__ == "__main__":
    index = "00055"
    fn_parse_agnostic = f"./tmp_parse_agnostic_{index}_00.png"
    fn_dense_pose = f"./tmp_dense_pose_{index}_00.jpg"
    fn_cloth = f"./tmp_cloth_{index}_00.jpg"
    fn_cloth_mask = f"./tmp_cloth_mask_{index}_00.jpg"
    img_pa = Image.open(fn_parse_agnostic)
    img_dp = Image.open(fn_dense_pose)
    img_c = Image.open(fn_cloth).convert('RGB')
    img_cm = Image.open(fn_cloth_mask)

    fine_width = 768
    img_pa = transforms.Resize(fine_width, interpolation=0)(img_pa)
    img_dp = transforms.Resize(fine_width, interpolation=2)(img_dp)
    img_c = transforms.Resize(fine_width, interpolation=2)(img_c)
    img_cm = transforms.Resize(fine_width, interpolation=0)(img_cm)
    np_cm = np.array(img_cm)
    np_cm = (np_cm >= 128).astype(np.float32)

    device = torch.device("cuda:0")

    tensor_pa = seg_to_onehot(np.array(img_pa))
    tensor_dp = hwc_to_nchw(np.array(img_dp))
    tensor_c = hwc_to_nchw(np.array(img_c))
    tensor_cm = hwc_to_nchw(np_cm, zero_center=False, norm=False)

    pfafn_infer = PFAFNImageInfer(
        device=device,
        ckpt_path="../../models/warp_viton.pth",
    )

    ret_dict = pfafn_infer.infer(
        tensor_pa, tensor_dp, tensor_c, tensor_cm
    )
    tensor_wc = ret_dict["warped_cloth"]
    tensor_wm = ret_dict["warped_mask"]

    np_wc = nchw_to_hwc(tensor_wc)
    np_wm = nchw_to_hwc(tensor_wm, zero_center=False)

    img_wc = Image.fromarray(np_wc)
    img_wm = Image.fromarray(np_wm.squeeze())

    img_wc.save("./tmp_wc.jpg")
    img_wm.save("./tmp_wm.jpg")

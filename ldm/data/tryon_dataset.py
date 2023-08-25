import json
import os
import glob
import cv2
import numpy as np
import torch

from tqdm import tqdm
from einops import rearrange
from torch.utils.data import Dataset


training_prompts = [
    "a photo",
    'the photo',
    'a dark photo',
    'a close-up photo',
    'a good photo',
    'a bright photo',
    'an illustration',
    'a rendering',
    'a cropped photo',
    'a rendition',
    'a depiction',
    'a photo of a person',
    'a rendering of a person',
    'a cropped photo of the person',
    'the photo of a person',
    'a photo of a clean person',
    'a photo of a dirty person',
    'a dark photo of the person',
    'a photo of my person',
    'a photo of the cool person',
    'a close-up photo of a person',
    'a bright photo of the person',
    'a cropped photo of a person',
    'a photo of the person',
    'a good photo of the person',
    'a photo of one person',
    'a close-up photo of the person',
    'a rendition of the person',
    'a photo of the clean person',
    'a rendition of a person',
    'a photo of a nice person',
    'a good photo of a person',
    'a photo of the nice person',
    'a photo of the small person',
    'a photo of the weird person',
    'a photo of the large person',
    'a photo of a cool person',
    'a photo of a small person',
    'an illustration of a person',
    'a rendering of a person',
    'a cropped photo of the person',
    'the photo of a person',
    'an illustration of a clean person',
    'an illustration of a dirty person',
    'a dark photo of the person',
    'an illustration of my person',
    'an illustration of the cool person',
    'a close-up photo of a person',
    'a bright photo of the person',
    'a cropped photo of a person',
    'an illustration of the person',
    'a good photo of the person',
    'an illustration of one person',
    'a close-up photo of the person',
    'a rendition of the person',
    'an illustration of the clean person',
    'a rendition of a person',
    'an illustration of a nice person',
    'a good photo of a person',
    'an illustration of the nice person',
    'an illustration of the small person',
    'an illustration of the weird person',
    'an illustration of the large person',
    'an illustration of a cool person',
    'an illustration of a small person',
    'a depiction of a person',
    'a rendering of a person',
    'a cropped photo of the person',
    'the photo of a person',
    'a depiction of a clean person',
    'a depiction of a dirty person',
    'a dark photo of the person',
    'a depiction of my person',
    'a depiction of the cool person',
    'a close-up photo of a person',
    'a bright photo of the person',
    'a cropped photo of a person',
    'a depiction of the person',
    'a good photo of the person',
    'a depiction of one person',
    'a close-up photo of the person',
    'a rendition of the person',
    'a depiction of the clean person',
    'a rendition of a person',
    'a depiction of a nice person',
    'a good photo of a person',
    'a depiction of the nice person',
    'a depiction of the small person',
    'a depiction of the weird person',
    'a depiction of the large person',
    'a depiction of a cool person',
    'a depiction of a small person',
]


class TryOnDataset(Dataset):
    def __init__(self, root: str,
                 mode: str = "train",
                 down_scale: float = 2.0,
                 ):
        self.root = root
        assert mode in ("train", "val", "test"), "[TryOnDataset] mode not supported!"
        self.mode = mode
        self.folder_name = "train" if mode in ("val",) else mode
        self.pairs_file = os.path.join(root, f"{self.folder_name}_pairs.txt")
        self.folder_path = os.path.join(root, self.folder_name)
        self.down_scale = down_scale

        self.data = self._load_dataset()

    def _load_dataset(self):
        with open(self.pairs_file, "r") as f:  # not used
            for line in f.readlines():
                f1, f2 = line.strip().split(" ")
        self.person_imgs = glob.glob(f"{self.folder_path}/image/*")
        self.person_denseposes = glob.glob(f"{self.folder_path}/image-densepose/*")
        self.person_openposes = glob.glob(f"{self.folder_path}/openpose_img/*")
        self.person_openpose_jsons = glob.glob(f"{self.folder_path}/openpose_json/*")
        self.person_parses = glob.glob(f"{self.folder_path}/image-parse-v3/*")
        self.agnostic_imgs = glob.glob(f"{self.folder_path}/agnostic-v3.2/*")
        self.agnostic_parses = glob.glob(f"{self.folder_path}/image-parse-agnostic-v3.2/*")
        self.cloth_imgs = glob.glob(f"{self.folder_path}/cloth/*")
        self.cloth_masks = glob.glob(f"{self.folder_path}/cloth-mask/*")

        data_list = []
        n = len(self.person_imgs)
        for i in range(n):
            data_list.append(
                {
                    "person_img": self.person_imgs[i],
                    "person_densepose": self.person_denseposes[i],
                    "person_openpose": self.person_openposes[i],
                    "person_openpose_json": self.person_openpose_jsons[i],
                    "person_parse": self.person_parses[i],
                    "agnostic_img": self.agnostic_imgs[i],
                    "agnostic_parse": self.agnostic_parses[i],
                    "cloth_img": self.cloth_imgs[i],
                    "cloth_mask": self.cloth_masks[i],
                }
            )

        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        person_fn = item["person_img"]
        openpose_fn = item["person_openpose"]
        agnostic_fn = item["agnostic_img"]
        cloth_fn = item["cloth_img"]
        cloth_mask_fn = item["cloth_mask"]

        prompt = training_prompts[np.random.randint(len(training_prompts))]

        person = cv2.imread(person_fn)
        openpose = cv2.imread(openpose_fn)
        agnostic = cv2.imread(agnostic_fn)
        cloth = cv2.imread(cloth_fn)
        cloth_mask = cv2.imread(cloth_mask_fn)

        h, w, c = person.shape
        th, tw = int(h / self.down_scale), int(w / self.down_scale)

        # Do not forget that OpenCV read images in BGR order.
        person = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
        openpose = cv2.cvtColor(openpose, cv2.COLOR_BGR2RGB)
        agnostic = cv2.cvtColor(agnostic, cv2.COLOR_BGR2RGB)
        person = cv2.resize(person, (tw, th), interpolation=cv2.INTER_LINEAR)
        openpose = cv2.resize(openpose, (tw, th), interpolation=cv2.INTER_LINEAR)
        agnostic = cv2.resize(agnostic, (tw, th), interpolation=cv2.INTER_LINEAR)
        cloth = cv2.cvtColor(cloth, cv2.COLOR_BGR2RGB)
        cloth = cv2.resize(cloth, (224, 224), interpolation=cv2.INTER_LINEAR)
        cloth_mask = cv2.cvtColor(cloth_mask, cv2.COLOR_BGR2RGB)
        cloth_mask = cv2.resize(cloth_mask, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Normalize source images to [0, 1].
        openpose = openpose.astype(np.float32) / 255.0
        cloth_mask = cloth_mask.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        person = (person.astype(np.float32) / 127.5) - 1.0
        agnostic = (agnostic.astype(np.float32) / 127.5) - 1.0
        cloth_masked = (cloth.astype(np.float32) * cloth_mask / 127.5) - 1.0

        return dict(jpg=person,  # reconstruction target
                    txt={"prompt": prompt, "cloth": cloth_masked},  # conditional inputs
                    hint=openpose,  # controlnet hint
                    agnostic=agnostic)  # concat with de-noised image
    
    @staticmethod
    def _nhwc_to_1hwrgb(x: torch.Tensor, is_zero_center: bool = True):
        x = x[0]
        if is_zero_center:
            x = (x + 1.) * 127.5
        else:
            x *= 255.
        x = x.cpu().numpy().astype(np.uint8)
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)  # to open-cv BGR format
        return x
    
    def get_batch0_snapshot(self, batch_dict: dict):
        return dict(
            jpg=self._nhwc_to_1hwrgb(batch_dict["jpg"]),
            txt={"prompt": batch_dict["txt"]["prompt"][0], 
                 "cloth": self._nhwc_to_1hwrgb(batch_dict["txt"]["cloth"])},
            hint=self._nhwc_to_1hwrgb(batch_dict["hint"], is_zero_center=False),
            agnostic=self._nhwc_to_1hwrgb(batch_dict["agnostic"])
        )
        

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    batch_size = 1
    dataset = TryOnDataset(
        root="/cfs/yuange/datasets/VTON-HD",
        mode="train",
    )
    dataloader = DataLoader(dataset, num_workers=2, batch_size=batch_size, shuffle=False)

    """ Check output files """
    snapshot_folder = "./snapshot"
    if os.path.exists(snapshot_folder):
        os.system(f"rm -r {snapshot_folder}")
    os.makedirs(snapshot_folder, exist_ok=True)
    max_index = 10

    for index, batch in enumerate(tqdm(dataloader)):
        if index >= max_index:
            break
        if index == 0:
            for key in batch.keys():
                val = batch[key]
                if isinstance(val, np.ndarray) or isinstance(val, torch.Tensor):
                    print(f"{key}:", val.shape)
                elif isinstance(val, list):
                    print(f"{key}:", len(val))
                elif isinstance(val, dict):
                    print(f"{key}:", val.keys())
                    for k in val.keys():
                        if isinstance(val[k], list):
                            print(f"|---{k}:", len(val[k]))
                        elif isinstance(val[k], torch.Tensor):
                            print(f"|---{k}:", val[k].shape)
                        else:
                            print(f"|---{k}:", type(val[k]))
                else:
                    print(f"{key}:", type(val))
        snapshot = dataset.get_batch0_snapshot(batch)
        fn = "id{:05d}".format(index)
        cv2.imwrite(os.path.join(snapshot_folder, "{}_{}.jpg".format(fn, "01person")), snapshot["jpg"])
        cv2.imwrite(os.path.join(snapshot_folder, "{}_{}.jpg".format(fn, "02cloth")), snapshot["txt"]["cloth"])
        cv2.imwrite(os.path.join(snapshot_folder, "{}_{}.jpg".format(fn, "03openpose")), snapshot["hint"])
        cv2.imwrite(os.path.join(snapshot_folder, "{}_{}.jpg".format(fn, "04agnostic")), snapshot["agnostic"])
    print(f"Snapshot files saved to: {snapshot_folder}")

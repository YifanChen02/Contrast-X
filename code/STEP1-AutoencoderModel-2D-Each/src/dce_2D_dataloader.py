# paired_loader.py
import csv
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import random


class TripletDCEDataset(Dataset):
    """
    Expects CSV with header:
    Dataset,Subject,ExamID,Slice,Split,DCE1,DCE2,DCE3

    Returns:
        {
          "x": torch.FloatTensor [3,H,W] in [0,1],
               (channel 0 = DCE1, channel 1 = DCE2, channel 2 = DCE3)
          "meta": dict with paths + IDs
        }
    """

    def __init__(
        self,
        csv_path: str,
        split: Optional[str] = None,       # "train" | "val" | "test" | None
        resolution: int = 256,             # resize before cropping
        spatial_size: Optional[int] = None,# random crop size (H,W)
        random_hflip: bool = False,
        check_files: bool = True,
    ):
        df = pd.read_csv(csv_path)

        # shuffle with fixed seed for reproducibility
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        df["Split"] = df["Split"].fillna("").str.lower()

        if split is not None and split != "test_all":
            df = df[df["Split"] == split.lower()]

        self.df = df.reset_index(drop=True)
        self.items = df.to_dict(orient="records")  # list of row dicts

        if not self.items:
            raise ValueError(f"No rows matched in CSV: {csv_path} (split={split})")

        self.resolution = resolution
        self.spatial_size = spatial_size
        self.random_hflip = random_hflip and (split == "train")
        self.split = split

        # resizing transform
        self.resize = transforms.Resize(resolution, interpolation=InterpolationMode.BICUBIC)

    def __len__(self):
        return len(self.items)

    def _load_gray(self, path: str) -> Image.Image:
        return Image.open(path).convert("L")

    def __getitem__(self, idx: int):
        it = self.items[idx]

        dce1_img = self._load_gray(it["DCE1"])
        dce2_img = self._load_gray(it["DCE2"])
        dce3_img = self._load_gray(it["DCE3"])


        # print("DCE1 path:", dce1_img.size)


        # resize all three consistently
        dce1_img = self.resize(dce1_img)
        dce2_img = self.resize(dce2_img)
        dce3_img = self.resize(dce3_img)

        # random crop (same coords for all)
        if self.spatial_size:
            i, j, h, w = transforms.RandomCrop.get_params(
                dce1_img, output_size=(self.spatial_size[0], self.spatial_size[1])
            )
            dce1_img = TF.crop(dce1_img, i, j, h, w)
            dce2_img = TF.crop(dce2_img, i, j, h, w)
            dce3_img = TF.crop(dce3_img, i, j, h, w)

        # flip (same coin for all)
        if self.random_hflip and random.random() < 0.5:
            dce1_img = TF.hflip(dce1_img)
            dce2_img = TF.hflip(dce2_img)
            dce3_img = TF.hflip(dce3_img)

        # to tensor [1,H,W]
        dce1 = TF.to_tensor(dce1_img)
        dce2 = TF.to_tensor(dce2_img)
        dce3 = TF.to_tensor(dce3_img)

        # stack into [3,H,W]
        x = torch.cat([dce1, dce2, dce3], dim=0)

        meta = {
            "Dataset": it["Dataset"],
            "Subject": it["Subject"],
            "ExamID":  it["ExamID"],
            "Slice":   it["Slice"],
            "Split":   it["Split"],
            "DCE1_path": it["DCE1"],
            "DCE2_path": it["DCE2"],
            "DCE3_path": it["DCE3"],
            "DCE1": dce1,
            "DCE2": dce2,
            "DCE3": dce3,
        }
        return meta #{"x": x, "meta": meta}


def create_triplet_dataloader(
    csv_path: str,
    split: Optional[str],
    batch_size: int = 8,
    resolution: int = 512,
    spatial_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    random_hflip: bool = True,
):
    if shuffle is None:
        shuffle = (split == "train")

    ds = TripletDCEDataset(
        csv_path=csv_path,
        split=split,
        resolution=resolution,
        spatial_size=spatial_size,
        random_hflip=random_hflip,
        check_files=True,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
    )
    return dl, ds


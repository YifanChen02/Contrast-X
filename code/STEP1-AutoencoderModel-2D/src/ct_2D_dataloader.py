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

class PairedCTDataset(Dataset):
    """
    Expects CSV with header:
    Dataset,Subject,ExamID,Slice,Splits,CT,CTC

    Returns:
        {
          "x": torch.FloatTensor [2,H,W] in [0,1],  (channel 0 = CT, channel 1 = CTC)
          "meta": dict
        }
    """

    def __init__(
        self,
        csv_path: str,
        split: Optional[str] = None,       # "train" | "val" | "test" | None
        resolution: int = 512,             # resize before cropping
        spatial_size: Optional[int] = None,# random crop size
        random_hflip: bool = False,
        check_files: bool = True,
    ):        

        df = pd.read_csv(csv_path)

        # shuffle with fixed seed if needed
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        df["Splits"] = df["Splits"].fillna("").str.lower()

        if split is not None and split != "test_all" :
            df = df[df["Splits"] == split.lower()]

        self.df = df.reset_index(drop=True)
        self.items = df.to_dict(orient="records")   # list of row dicts


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
        ct_img  = self._load_gray(it["CT"])
        ctc_img = self._load_gray(it["CTC"])

        # random crop (same coords for both)
        if self.spatial_size:
            i, j, h, w = transforms.RandomCrop.get_params(
                ct_img, output_size=(self.spatial_size[0], self.spatial_size[1])
            )
            ct_img  = TF.crop(ct_img, i, j, h, w)
            ctc_img = TF.crop(ctc_img, i, j, h, w)

        # flip (same coin for both)
        if self.random_hflip and random.random() < 0.5:
            ct_img  = TF.hflip(ct_img)
            ctc_img = TF.hflip(ctc_img)

        # to tensor (0–1 range)
        ct  = TF.to_tensor(ct_img)   # [1,H,W], float32 [0,1]
        ctc = TF.to_tensor(ctc_img)

        # x = torch.cat([ct, ctc], dim=0)   # [2,H,W]
    
        meta = {
            "Dataset": it["Dataset"],
            "Subject": it["Subject"],
            "ExamID":  it["ExamID"],
            "Slice":   it["Slice"],
            "Splits":  it["Splits"],
            "CT_path": it["CT"],
            "CTC_path":it["CTC"],
            "CT": ct, "CTC": ctc
        }
        return meta


def create_paired_dataloader(
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

    ds = PairedCTDataset(
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

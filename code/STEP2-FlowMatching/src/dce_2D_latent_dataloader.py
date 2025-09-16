# paired_loader.py
import csv
from pathlib import Path
from typing import Optional
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import Image
from torchvision import transforms




class PairedLatentDataset(Dataset):
    """
    Expects CSV with header:
    Dataset,Subject,ExamID,Slice,Split,DCE1,DCE2,DCE3

    Instead of loading .jpg directly, this dataset loads
    latent tensors saved as OUT_ROOT/.../DCE1.npz etc.
    """

    def __init__(
        self,
        csv_path: str,
        data_root: str,
        out_root: str,
        target_dataset: list = None,
        split: Optional[str] = None,
        spatial_size: Optional[int] = None,
        random_hflip: bool = False,
    ):
        df = pd.read_csv(csv_path)
        # df = df[df["Dataset"].apply(lambda x: x.split("_")[0]).isin(target_dataset)]
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df["Split"] = df["Split"].fillna("").str.lower()

        if split is not None and split != "test_all":
            df = df[df["Split"] == split.lower()]

        self.df = df.reset_index(drop=True)
        self.items = df.to_dict(orient="records")

        if not self.items:
            raise ValueError(f"No rows matched in CSV: {csv_path} (split={split})")

        self.data_root = Path(data_root)
        self.out_root = Path(out_root)
        self.spatial_size = spatial_size
        self.random_hflip = random_hflip and (split == "train")
        self.split = split

    def __len__(self):
        return len(self.items)

    def _latent_path(self, img_path: str, name: str):
        img_path = Path(img_path)
        rel_dir = img_path.relative_to(self.data_root).parent
        out_dir = self.out_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{name}.npz"

    def _load_gray(self, path: str) -> Image.Image:
        return Image.open(path).convert("L")

    def _load_latent(self, path):
        arr = np.load(str(path), allow_pickle=False)["data"]
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int):
        it = self.items[idx]

        # latent paths
        dce1_path = self._latent_path(it["DCE1"], "DCE_1")
        dce2_path = self._latent_path(it["DCE2"], "DCE_13")
        dce3_path = self._latent_path(it["DCE3"], "DCE_123")    # DCE_123

        # load images
        dce1_img = TF.to_tensor(self._load_gray(it["DCE1"]))
        dce2_img = TF.to_tensor(self._load_gray(it["DCE2"]))
        dce3_img = TF.to_tensor(self._load_gray(it["DCE3"]))

        if self.spatial_size:
            i, j, h, w = transforms.RandomCrop.get_params(dce1_img, output_size=self.spatial_size)
            dce1_img = TF.crop(dce1_img, i, j, h, w)
            dce2_img = TF.crop(dce2_img, i, j, h, w)
            dce3_img = TF.crop(dce3_img, i, j, h, w)

        # load latents
        dce1 = self._load_latent(dce1_path)
        dce2 = self._load_latent(dce2_path)
        dce3 = self._load_latent(dce3_path)

        # random flip
        # if self.random_hflip and random.random() < 0.5:
        #     if dce1.ndim == 2:
        #         dce1 = TF.hflip(dce1.unsqueeze(0)).squeeze(0)
        #         dce2 = TF.hflip(dce2.unsqueeze(0)).squeeze(0)
        #         dce3 = TF.hflip(dce3.unsqueeze(0)).squeeze(0)
        #     else:
        #         dce1 = TF.hflip(dce1)
        #         dce2 = TF.hflip(dce2)
        #         dce3 = TF.hflip(dce3)

        meta = {
            "Dataset": it["Dataset"],
            "Subject": it["Subject"],
            "ExamID":  it["ExamID"],
            "Slice":   it["Slice"],
            "Split":  it["Split"],

            "DCE1_path": str(dce1_path),
            "DCE2_path": str(dce2_path),
            "DCE3_path": str(dce3_path),

            "DCE1_img": dce1_img,
            "DCE2_img": dce2_img,
            "DCE3_img": dce3_img,

            "DCE_1": dce1.float(),
            "DCE_13": dce2.float(),
            "DCE_123": dce3.float(),
        }
        return meta


def create_paired_dataloader(
    csv_path: str,
    data_root: str,
    out_root: str,
    target_dataset: list,
    split: Optional[str],
    batch_size: int = 8,
    spatial_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    random_hflip: bool = True,
):
    if shuffle is None:
        shuffle = (split == "train")

    ds = PairedLatentDataset(
        csv_path=csv_path,
        data_root=data_root,
        out_root=out_root,
        target_dataset=target_dataset,
        split=split,
        spatial_size=spatial_size,
        random_hflip=random_hflip,
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

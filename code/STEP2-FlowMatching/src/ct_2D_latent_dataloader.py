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


target_dataset = ["Adrenal"]
# target_dataset = ["Adrenal", "Bladder", "Lung", "Stomach", "Uterus"]


class PairedLatentDataset(Dataset):
    """
    Expects CSV with header:
    Dataset,Subject,ExamID,Slice,Splits,CT,CTC

    Instead of loading CT.jpg/CTC.jpg, this dataset loads
    latent tensors saved as OUT_ROOT/.../CT.npz and CTC.npz.
    """

    def __init__(
        self,
        csv_path: str,
        data_root: str,
        out_root: str,
        target_dataset: list = target_dataset,
        split: Optional[str] = None,       # "train" | "val" | "test" | None
        spatial_size: Optional[int] = None,
        random_hflip: bool = False,
    ):
        df = pd.read_csv(csv_path)

        df = df[df["Dataset"].apply(lambda x: x.split("_")[0]).isin(target_dataset)]

        # shuffle with fixed seed if needed
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df["Splits"] = df["Splits"].fillna("").str.lower()

        if split is not None and split != "test_all":
            df = df[df["Splits"] == split.lower()]

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

        # ---- get "patient_path" you asked for ----
        rel_dir = img_path.relative_to(self.data_root).parent
        patient_path = str(rel_dir)  # "Stomach_Colon_Liver_Pancreas_CT_train_val_test/train/HCC_023_1999-12-24_CT/slice_033"

        # ---- make an output dir that mirrors the dataset tree ----
        out_dir = self.out_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        destpath = out_dir / f"{name}.npz" # 'CT.npz'
        return destpath


    def _load_gray(self, path: str) -> Image.Image:
        return Image.open(path).convert("L")


    def _load_latent(self, path):
        if path is None:
            raise ValueError("Tried to load latent from None path")
        arr = np.load(str(path), allow_pickle=False)["data"]
        return torch.from_numpy(arr)
    

    def __getitem__(self, idx: int):
        it = self.items[idx]

        ct_path  = self._latent_path(it["CT"], "CT")
        ctc_path = self._latent_path(it["CTC"], "CT_CTC")

        ct_img  = self._load_gray(it["CT"])
        ctc_img = self._load_gray(it["CTC"])
        ct_img  = TF.to_tensor(ct_img)   # [1,H,W], float32 [0,1]
        ctc_img = TF.to_tensor(ctc_img)

        if self.spatial_size:
            i, j, h, w = transforms.RandomCrop.get_params(
                ct_img, output_size=(self.spatial_size[0], self.spatial_size[1])
            )
            ct_img  = TF.crop(ct_img, i, j, h, w)
            ctc_img = TF.crop(ctc_img, i, j, h, w)

        ct  = self._load_latent(ct_path)
        ctc = self._load_latent(ctc_path)

        # flip (same coin for both) if image-like
        if self.random_hflip and random.random() < 0.5 and ct.ndim >= 2:
            if ct.ndim == 2:  # [H,W]
                ct  = TF.hflip(ct.unsqueeze(0)).squeeze(0)
                ctc = TF.hflip(ctc.unsqueeze(0)).squeeze(0)
            else:  # [C,H,W]
                ct  = TF.hflip(ct)
                ctc = TF.hflip(ctc)


        meta = {
            "Dataset": it["Dataset"],
            "Subject": it["Subject"],
            "ExamID":  it["ExamID"],
            "Slice":   it["Slice"],
            "Splits":  it["Splits"],
            "CT_path": str(ct_path),
            "CTC_path": str(ctc_path),
            "CT_img": ct_img,
            "CTC_img": ctc_img,

            "CT": ct.float(),
            "CTC": ctc.float(),
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

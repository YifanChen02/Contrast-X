#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image
from math import log10

# ----------------------------
# Config
# ----------------------------
DATA_DIR = Path("/home/yifan/data")

CT_SUBFOLDERS = [
    "Adrenal_CT_train_val_test",
    "Bladder_Kidney_CT_train_val_test",
    "Lung_CT_train_val_test",
    "Stomach_Colon_Liver_Pancreas_CT_train_val_test",
    "Uterus_Ovary_CT_train_val_test",
]

Brain_MR_SUBFOLDERS = [
    "Breast_MR_train_val_test",
    # "Brain_MR_train_val_test",
]

CT_KEYS = {
    "ct": re.compile(r"(?:^|[_\-\.])ct(?:[_\-\.]|$)", re.IGNORECASE),
    "ctc": re.compile(r"(?:^|[_\-\.])ctc(?:[_\-\.]|$)", re.IGNORECASE),
}

# ----------------------------
# Organ-wise HU clipping
# ----------------------------
HU_CLIP_BY_FOLDER = {
    "Lung_CT_train_val_test": (-1150, 350),           # lung
    "Adrenal_CT_train_val_test": (-140, 260),         # abdominal soft tissue
    "Bladder_Kidney_CT_train_val_test": (-135, 215),  # kidney
    "Stomach_Colon_Liver_Pancreas_CT_train_val_test": (-50, 150),  # liver-focused
    "Uterus_Ovary_CT_train_val_test": (-140, 260),    # pelvic/abdominal
}

DEFAULT_HU_CLIP = (-100, 300)  # fallback for unknown folders

NIFTI_EXTS = {".nii", ".nii.gz"}

# ----------------------------
# Helpers
# ----------------------------
def load_nifti(path: Path) -> np.ndarray:
    """Load NIfTI and return float32 array (nibabel applies slope/intercept -> HU)."""
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32)

def get_modality(name: str) -> str | None:
    for key, pat in CT_KEYS.items():
        if pat.search(name):
            return key.upper()
    return None

def hu_clip_to_unit(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip HU to [lo,hi] and scale to [0,1] float."""
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip(arr, lo, hi)
    return (arr - lo) / (hi - lo + 1e-8)

def hu_clip_to_uint8(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip HU to [lo,hi] and scale to uint8 [0,255]."""
    unit = hu_clip_to_unit(arr, lo, hi)
    return (unit * 255.0).astype(np.uint8)

def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two arrays in [0,1]."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * log10(1.0 / np.sqrt(mse))

# ----------------------------
# Core: save CT/CTC pair with HU windowing
# ----------------------------
def save_ct_ctc_pair(ct_path: Path, ctc_path: Path, out_root: Path, vis_root: Path):
    ct_vol = load_nifti(ct_path)
    ctc_vol = load_nifti(ctc_path)

    assert ct_vol.shape == ctc_vol.shape, f"Shape mismatch: {ct_path} vs {ctc_path}"

    case_name   = ct_path.stem.replace(".nii", "")
    dataset_name = ct_path.parents[3].name   # e.g. Uterus_Ovary_CT_train_val_test
    split_name   = ct_path.parents[2].name   # train / val / test
    case_id      = case_name                 # e.g. C3N-00866_2000-03-05

    # Pick HU clip by dataset
    lo, hi = HU_CLIP_BY_FOLDER.get(dataset_name, DEFAULT_HU_CLIP)

    out_case_dir = out_root / dataset_name / split_name / case_id
    out_case_dir.mkdir(parents=True, exist_ok=True)

    num_slices = ct_vol.shape[-1]
    psnr_vals = []

    for idx in range(num_slices):
        ct_slice  = ct_vol[..., idx]
        ctc_slice = ctc_vol[..., idx]

        # HU -> [0,1] for metrics / checks
        ct_unit  = hu_clip_to_unit(ct_slice, lo, hi)
        ctc_unit = hu_clip_to_unit(ctc_slice, lo, hi)

        # Vacant slice guard (after proper HU windowing)
        if np.std(ct_unit) < 1e-2 or np.std(ctc_unit) < 1e-2:
            print(f"[WARNING] Slice may be vacant: {dataset_name}/{split_name}/{case_id}/slice_{idx:03d}")
            continue

        # PSNR on HU-windowed [0,1]
        psnr_vals.append(np.clip(psnr(ct_unit, ctc_unit), 0, 100))

        # Save uint8 JPGs
        slice_dir = out_case_dir / f"slice_{idx:03d}"
        slice_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(hu_clip_to_uint8(ct_slice, lo, hi)).save(slice_dir / "CT.jpg", quality=95)
        Image.fromarray(hu_clip_to_uint8(ctc_slice, lo, hi)).save(slice_dir / "CTC.jpg", quality=95)

    # Save mid-slice previews
    mid_idx = num_slices // 2
    vis_dir = vis_root / dataset_name / split_name / case_id
    vis_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(hu_clip_to_uint8(ct_vol[..., mid_idx],  lo, hi)).save(vis_dir / "CT_mid.jpg",  quality=95)
    Image.fromarray(hu_clip_to_uint8(ctc_vol[..., mid_idx], lo, hi)).save(vis_dir / "CTC_mid.jpg", quality=95)

    if psnr_vals:
        print(f"[{dataset_name}/{split_name}/{case_id}] Avg PSNR (CT vs CTC): {np.mean(psnr_vals):.3f}")
    else:
        print(f"[{dataset_name}/{split_name}/{case_id}] No valid slices for PSNR.")

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA_DIR, help="Root dataset folder")
    parser.add_argument("--output", type=Path, required=True, help="Output 2D dataset root")
    parser.add_argument("--vis", type=Path, required=True, help="Output 2Dvis folder")
    args = parser.parse_args()

    for sub in CT_SUBFOLDERS:
        subdir = args.input / sub
        if not subdir.exists():
            continue
        print(f"[INFO] Processing {subdir}")

        # Group files by case (CT + CTC pair)
        case_files = {}
        for vol_path in subdir.rglob("*"):
            if not any(str(vol_path).endswith(ext) for ext in NIFTI_EXTS):
                continue
            stem = vol_path.stem.replace(".nii", "")
            base_id = stem.split("_CT")[0].split("_CTC")[0]
            modality = get_modality(vol_path.name)
            if modality is None:
                continue
            case_files.setdefault(base_id, {})[modality] = vol_path

        # Process pairs
        for case_id, files in case_files.items():
            if "CT" in files and "CTC" in files:
                try:
                    save_ct_ctc_pair(files["CT"], files["CTC"], args.output, args.vis)
                except Exception as e:
                    print(f"[ERROR] Failed on {case_id}: {e}")
            else:
                print(f"[WARN] Missing CT/CTC pair for case {case_id}")

if __name__ == "__main__":
    main()

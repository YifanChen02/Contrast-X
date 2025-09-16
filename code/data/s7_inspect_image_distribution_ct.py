import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.exposure import match_histograms

# -------------------
# Config
# -------------------
csv_path = "/home/hao/repo/FM-translation/code/data/files/2D_CT_pair.csv"
save_dir = "residual_plots"
os.makedirs(save_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)
print(df.head())

ct_col = "CT"
ctc_col = "CTC"

# -------------------
# Number of samples to plot
# -------------------
num_samples = min(6, len(df))  # at most 6

# Create figures
fig_max, axes_max = plt.subplots(2, 3, figsize=(15, 10))
axes_max = axes_max.flatten()

fig_std, axes_std = plt.subplots(2, 3, figsize=(15, 10))
axes_std = axes_std.flatten()

fig_match, axes_match = plt.subplots(2, 3, figsize=(15, 10))
axes_match = axes_match.flatten()



def normalize_percentile(img2d, lower=1, upper=99):
    """
    Percentile-based normalization for 2D MRI slices.
    
    Args:
        img2d (np.ndarray): 2D MRI image (grayscale).
        lower (float): lower percentile for clipping (default: 1).
        upper (float): upper percentile for clipping (default: 99).
        
    Returns:
        np.ndarray: normalized image in [0,1].
    """
    # Convert to float
    img = img2d.astype(np.float32)

    # Get percentile values
    p_low, p_high = np.percentile(img, (lower, upper))

    # Clip to percentiles
    img = np.clip(img, p_low, p_high)

    # Scale to [0,1]
    img = (img - p_low) / (p_high - p_low + 1e-8)

    return img



for i in range(num_samples):
    ct_path = df.loc[i, ct_col]
    ctc_path = df.loc[i, ctc_col]

    # Read grayscale images
    CT = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    CTC = cv2.imread(ctc_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # --- Method 1: Per-image min-max normalization ---
    CT_minmax = (CT - CT.min()) / (CT.max() - CT.min() + 1e-8)
    CTC_minmax = (CTC - CTC.min()) / (CTC.max() - CTC.min() + 1e-8)
    # CT_minmax = CT_minmax - CT_minmax.mean()  # center around 0
    # CTC_minmax = CTC_minmax - CTC_minmax.mean()

    

    residual_minmax = CTC_minmax - CT_minmax

    print("Stat of residual (minmax):", residual_minmax.min(), residual_minmax.max(), residual_minmax.mean(), residual_minmax.std())


    im1 = axes_max[i].imshow(residual_minmax, cmap="bwr")
    axes_max[i].set_title(f"Residual {i} (min–max)")
    axes_max[i].axis("off")

    # --- Method 2: Z-score standardization ---
    CT_std  = (CT - CT.mean()) / (CT.std() + 1e-8)
    CTC_std = (CTC - CTC.mean()) / (CTC.std() + 1e-8)
    residual_std = CTC_std - CT_std
    # residual_std = residual_std  # * CTC.std()  + CTC.mean()  # scale back to original range
    print("Stat of residual (std):", residual_std.min(), residual_std.max(), residual_std.mean(), residual_std.std())

    im2 = axes_std[i].imshow(residual_std, cmap="bwr", vmin=-3, vmax=3)
    axes_std[i].set_title(f"Residual {i} (z-score)")
    axes_std[i].axis("off")

    # --- Method 3: Histogram matching ---
    CT = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    CTC = cv2.imread(ctc_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    CTC = normalize_percentile(CTC, 0, 100)
    CT  = normalize_percentile(CT, 0, 100)
    residual_match = CTC - CT
    print("Stat of residual (hist match):", residual_match.min(), residual_match.max(), residual_match.mean(), residual_match.std())

    im3 = axes_match[i].imshow(residual_match, cmap="bwr")
    axes_match[i].set_title(f"Residual {i} (hist match)")
    axes_match[i].axis("off")

# -------------------
# Save figures
# -------------------
fig_max.colorbar(im1, ax=axes_max.tolist(), shrink=0.6, label="Residual intensity")
fig_max.tight_layout()
path = os.path.join(save_dir, "residual_grid_minmax.png")
fig_max.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig_max)
print(f"Saved {path}")

fig_std.colorbar(im2, ax=axes_std.tolist(), shrink=0.6, label="Residual intensity")
fig_std.tight_layout()
path = os.path.join(save_dir, "residual_grid_std.png")
fig_std.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig_std)
print(f"Saved {path}")

fig_match.colorbar(im3, ax=axes_match.tolist(), shrink=0.6, label="Residual intensity")
fig_match.tight_layout()
path = os.path.join(save_dir, "residual_grid_match.png")
fig_match.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig_match)
print(f"Saved {path}")



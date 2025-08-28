import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- CONFIG ---
#     cd /home/hao/repo/FM-translation/code/data/Gen_FM_Fig  ;   conda activate progression

IMG1_PATH = "./Fig/CT.png"   # replace with your source image
IMG2_PATH = "./Fig/CTC.png"  # replace with your target image
RESULTS_DIR = "results"
STEPS = 5                 # number of intermediate frames

# --- PREPARE ---
os.makedirs(RESULTS_DIR, exist_ok=True)


# Load images
img1 = cv2.imread(IMG1_PATH)
img2 = cv2.imread(IMG2_PATH)
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))  # match sizes

# Convert to grayscale for flow estimation
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


ut = gray2.astype(np.float32) - gray1.astype(np.float32)

# Normalize for visualization (0..1)
ut_norm = (ut - ut.min()) / (ut.max() - ut.min() + 1e-8)

# Save ut as an image
cmaps = ["seismic", "coolwarm", "bwr", "viridis", "plasma", "inferno",
         "gray", "binary", "Greys"]

for cmap in cmaps:
    ut_path = os.path.join(RESULTS_DIR, f"ut_{cmap}.png")
    plt.imsave(ut_path, ut_norm, cmap=cmap)
    print(f"Saved {ut_path}")




# Compute optical flow (Farneback)
flow = cv2.calcOpticalFlowFarneback(gray1, gray2,
                                    None,
                                    pyr_scale=0.5,
                                    levels=3,
                                    winsize=15,
                                    iterations=3,
                                    poly_n=5,
                                    poly_sigma=1.2,
                                    flags=0)

# --- INTERPOLATION ---
h, w = gray1.shape
grid_y, grid_x = np.mgrid[0:h, 0:w]

for i, alpha in enumerate(np.linspace(0, 1, STEPS)):
    # Interpolate pixel positions using velocity field
    dx = flow[..., 0] * alpha
    dy = flow[..., 1] * alpha

    map_x = (grid_x + dx).astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)

    # Warp image1 towards image2
    warped = cv2.remap(img1, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    # Blend between warped source and target for smoothness
    blended = cv2.addWeighted(warped, 1 - alpha, img2, alpha, 0)

    # Save frame
    out_path = os.path.join(RESULTS_DIR, f"frame_{i:03d}.png")
    cv2.imwrite(out_path, blended)

    # Optional visualization at checkpoints
    if i % (STEPS // 5) == 0 or i == STEPS - 1:
        plt.figure(figsize=(5,5))
        plt.title(f"Step {i}/{STEPS-1}")
        plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()


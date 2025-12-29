import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from src.dce_2D_latent_dataloader import create_paired_dataloader
from utils import args

# -------------------
# Config
# -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_key = args.input_modality

key_to_load = []
key_to_load.extend(image_key)

data_root = args.data_dir
latent_root = args.latent_dir
spatial_size = [256, 256]

target_dataset = ["Adrenal"]

# --- Here: change modalities ---
latent_key = "DCE_Full"       # target latent
broken_latent_key = "DCE_1"   # reference latent

plot_item_num = 300
tag = "DCE1_vs_full"

# -------------------
# Create loaders
# -------------------
train_loader, train_ds = create_paired_dataloader(
    csv_path=args.dataset_csv, target_dataset=target_dataset,
    data_root=data_root, out_root=latent_root,
    split="train", batch_size=args.batch_size, spatial_size=spatial_size)

# -------------------
# Collect samples
# -------------------
all_x0, all_x1 = [], []
progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
progress_bar.set_description("Collecting data for t-SNE")

# Scale factors
rate_num = 25
z_list = [train_loader.dataset[i][latent_key] for i in range(rate_num)]
z = torch.stack(z_list, dim=0)
scale_factor_full = 1 / torch.std(z)

z_list = [train_loader.dataset[i][broken_latent_key] for i in range(rate_num)]
z = torch.stack(z_list, dim=0)
scale_factor_broken = 1 / torch.std(z)
scale_factor = [scale_factor_broken, scale_factor_full]

for step, batch in progress_bar:
    x0 = batch[broken_latent_key].to(DEVICE).clone() * scale_factor[0]
    x1 = batch[latent_key].to(DEVICE).clone() * scale_factor[1]

    # standardize per-sample
    x0 = x0 / (x0.std() + 1e-8)
    x1 = x1 / (x1.std() + 1e-8)

    B = x0.shape[0]
    all_x0.append(x0.view(B, -1).detach().cpu())
    all_x1.append(x1.view(B, -1).detach().cpu())

    if step > plot_item_num:
        break

x0_flat = torch.cat(all_x0, dim=0).numpy()
x1_flat = torch.cat(all_x1, dim=0).numpy()
B_total = x0_flat.shape[0]

X = np.concatenate([x0_flat, x1_flat], axis=0)
labels = np.array([0] * B_total + [1] * B_total)

# -------------------
# Run t-SNE
# -------------------
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
X_2d = tsne.fit_transform(X)
X0_2d, X1_2d = X_2d[:B_total], X_2d[B_total:]

# -------------------
# Save plots
# -------------------
save_dir = "tsne_plots"
os.makedirs(save_dir, exist_ok=True)

# Pair-link t-SNE plot
plt.figure(figsize=(8, 6))
plt.scatter(X0_2d[:, 0], X0_2d[:, 1], c="blue", alpha=0.6, label=broken_latent_key)
plt.scatter(X1_2d[:, 0], X1_2d[:, 1], c="red", alpha=0.6, label=latent_key)
for i in range(B_total):
    plt.plot([X0_2d[i, 0], X1_2d[i, 0]], [X0_2d[i, 1], X1_2d[i, 1]],
             c="gray", alpha=0.3, linewidth=0.5)
plt.legend()
plt.title(f"t-SNE visualization of {broken_latent_key}–{latent_key} pairs")

save_path = os.path.join(save_dir, f"tsne_pairs_{tag}_dce.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved pair-link plot to {save_path}")

# Distance histogram
dists = np.linalg.norm(X0_2d - X1_2d, axis=1)
plt.figure(figsize=(6, 4))
plt.hist(dists, bins=30, color="purple", alpha=0.7)
plt.xlabel(f"t-SNE distance between {broken_latent_key}–{latent_key} pairs")
plt.ylabel("Count")
plt.title("Pairwise distance distribution")

save_path = os.path.join(save_dir, f"tsne_distances_{tag}_dce.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved distance histogram to {save_path}")

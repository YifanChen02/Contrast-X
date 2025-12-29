import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from src.ct_2D_latent_dataloader import create_paired_dataloader
from utils import args

# -------------------
# Config
# -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dimension = 2
image_key = args.input_modality

key_to_load  = []         
key_to_load.extend(image_key)

data_root    = args.data_dir
latent_root  = args.latent_dir
spatial_size = [256, 256]

target_dataset = ["Adrenal"]
# target_dataset = ["Adrenal", "Bladder", "Lung", "Stomach", "Uterus"]


latent_key     = "CTC"
file_key       = "CT_path"  # CTC_path
broken_latent_key = "CT"

plot_item_num = 1000

tag="different_scale"
tag="own_scale"


# -------------------
# Create loaders
# -------------------
train_loader, train_ds = create_paired_dataloader(
    csv_path=args.dataset_csv, target_dataset=target_dataset,
    data_root=data_root, out_root=latent_root,
    split="train", batch_size=args.batch_size, spatial_size=spatial_size)

test_loader, test_ds = create_paired_dataloader(
    csv_path=args.dataset_csv, target_dataset=target_dataset,
    data_root=data_root, out_root=latent_root,
    split="test", batch_size=4 * args.batch_size, spatial_size=spatial_size)


loader = train_loader  # or train_loader

# -------------------
# Example: collect x0/x1 and visualize with t-SNE
# -------------------

all_x0, all_x1 = [], []

progress_bar = tqdm(enumerate(loader), total=len(loader))
progress_bar.set_description("Collecting data for t-SNE")



scale_factor = []
rate_num = 25

z_list = [train_loader.dataset[i][latent_key] for i in range(rate_num)]
z = torch.stack(z_list, dim=0)  # Stack into a single tensor
scale_factor_full = 1 / torch.std(z)  

z_list = [train_loader.dataset[i][broken_latent_key] for i in range(rate_num)]
z = torch.stack(z_list, dim=0)  # Stack into a single tensor
scale_factor_broken = 1 / torch.std(z)  
scale_factor = [scale_factor_broken, scale_factor_full]





for step, batch in progress_bar:
    x0 = batch[broken_latent_key].to(DEVICE).clone() * scale_factor[1]
    x1 = batch[latent_key].to(DEVICE).clone() * scale_factor[1]  # scale_factor[0] #


    # std = ()
    # x0 = (x0 ) / (x0.std() + 1e-8)
    # x1 = (x1 ) / (x1.std() + 1e-8)

    B = x0.shape[0]
    all_x0.append(x0.view(B, -1).detach().cpu())
    all_x1.append(x1.view(B, -1).detach().cpu())

    if step > plot_item_num:   # limit number of batches for speed
        break


x0_flat = torch.cat(all_x0, dim=0).numpy()
x1_flat = torch.cat(all_x1, dim=0).numpy()
B_total = x0_flat.shape[0]

X = np.concatenate([x0_flat, x1_flat], axis=0)
labels = np.array([0]*B_total + [1]*B_total)

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X)

# X_2d = tsne.fit_transform(X)

X0_2d, X1_2d = X_2d[:B_total], X_2d[B_total:]



import os

# Make sure folder exists
save_dir = "tsne_plots"
os.makedirs(save_dir, exist_ok=True)





# -------------------
# Pair-link t-SNE plot
# -------------------
plt.figure(figsize=(8, 6))
plt.scatter(X0_2d[:,0], X0_2d[:,1], c="blue", alpha=0.6, label="x0")
plt.scatter(X1_2d[:,0], X1_2d[:,1], c="red", alpha=0.6, label="x1")
for i in range(B_total):
    plt.plot([X0_2d[i,0], X1_2d[i,0]], [X0_2d[i,1], X1_2d[i,1]],
             c="gray", alpha=0.3, linewidth=0.5)
plt.legend()
plt.title("t-SNE visualization of x0–x1 pairs")

save_path = os.path.join(save_dir, f"tsne_pairs_{tag}.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved pair-link plot to {save_path}")

# -------------------
# Distance histogram
# -------------------
dists = np.linalg.norm(x1_flat - x0_flat, axis=1)
plt.figure(figsize=(6,4))
plt.hist(dists, bins=30, color="purple", alpha=0.7)
plt.xlabel("t-SNE distance between x0–x1 pairs")
plt.ylabel("Count")
plt.title("Pairwise distance distribution")

save_path = os.path.join(save_dir, f"tsne_distances_{tag}.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved distance histogram to {save_path}")




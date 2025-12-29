import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import OrderedDict

from src.dce_2D_latent_dataloader import create_paired_dataloader
from src.MM_AE import AutoencoderKL_multi_encoder
from utils import args

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

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

latent_key = "DCE_123"       # target latent
broken_latent_key = "DCE_1"   # reference latent

plot_item_num = 6   # how many samples to visualize & save
output_root = "./interp_results"
os.makedirs(output_root, exist_ok=True)

# -------------------
# Create loaders
# -------------------
train_loader, train_ds = create_paired_dataloader(
    csv_path=args.dataset_csv, target_dataset=target_dataset,
    data_root=data_root, out_root=latent_root,
    split="test_all", batch_size=args.batch_size, spatial_size=spatial_size)

# -------------------
# Collect samples
# -------------------
all_x0, all_x1 = [], []
progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
progress_bar.set_description("Collecting data for interpolation")

# Scale factors
rate_num = 25
z_list = [train_loader.dataset[i][latent_key] for i in range(rate_num)]
z = torch.stack(z_list, dim=0)
scale_factor_full = 1 / torch.std(z)

z_list = [train_loader.dataset[i][broken_latent_key] for i in range(rate_num)]
z = torch.stack(z_list, dim=0)
scale_factor_broken = 1 / torch.std(z)
scale_factor = [scale_factor_broken, scale_factor_full]



# -------------------
# Define Autoencoder
# -------------------
def define_2DAE(in_channels=3, out_channels=2, latent_channels=16):
    block_out_channels = (128, 256)
    layers_per_block = 3
    n_blocks = len(block_out_channels)

    vae = AutoencoderKL_multi_encoder(
        num_encode=in_channels,
        in_channels=1, out_channels=out_channels, latent_channels=latent_channels,
        block_out_channels=block_out_channels,
        down_block_types=("DownEncoderBlock2D",) * n_blocks,
        up_block_types=("UpDecoderBlock2D",) * n_blocks,
        layers_per_block=layers_per_block,
        norm_num_groups=32
    )
    return vae

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    return new_state_dict

print("Setting up Autoencoder model...")
autoencoder = define_2DAE(
    in_channels=3,
    out_channels=3,
    latent_channels=16
).to(DEVICE).float()

print("Finish setting up...")
try:
    weight = torch.load(args.aekl_ckpt)
    weight = remove_module_prefix(weight)
    autoencoder.load_state_dict(weight)
except FileNotFoundError:
    print(f"File {args.aekl_ckpt} not found, using random initialization for autoencoder.")

autoencoder.to(DEVICE)
autoencoder.eval()


for step, batch in progress_bar:
    x0 = batch[broken_latent_key].to(DEVICE).clone() * scale_factor[0]
    x1 = batch[latent_key].to(DEVICE).clone() * scale_factor[1]

    all_x0.append(x0.detach().cpu())
    all_x1.append(x1.detach().cpu())

    if step > plot_item_num:
        break


# -------------------
# Interpolation & Results
# -------------------
n_steps = 6
alphas = torch.linspace(0, 1, n_steps).to(DEVICE)

fig, axes = plt.subplots(plot_item_num, n_steps + 2, figsize=(3*(n_steps+1), 3*plot_item_num))

for sample_idx in range(min(plot_item_num, len(all_x0))):
    x0 = all_x0[sample_idx][0].unsqueeze(0).to(DEVICE)
    x1 = all_x1[sample_idx][0].unsqueeze(0).to(DEVICE)

    interpolations = []
    with torch.no_grad():
        for a in alphas:
            z = (1 - a) * x0 + a * x1
            z = z.detach().clone()
            x_decoded = autoencoder.decode(z / scale_factor[1]).sample.cpu()
            # print("x_decoded = ", x_decoded.shape)  # 1,3
            x_decoded = x_decoded[:, 2:3]

            interpolations.append(x_decoded)
    interpolations = torch.cat(interpolations, dim=0)

    # Show interpolation images in row
    for i in range(n_steps):
        img = interpolations[i][0].numpy()
        axes[sample_idx, i].imshow(img, cmap="gray")
        axes[sample_idx, i].set_title(f"α={alphas[i]:.2f}" if sample_idx == 0 else "")
        axes[sample_idx, i].axis("off")

    # Compute SSIM & PSNR
    x1_decoded = interpolations[-1][0].numpy()
    metrics_ssim, metrics_psnr = [], []
    for i in range(n_steps):
        x_interp = interpolations[i][0].numpy()
        x1_norm = (x1_decoded - x1_decoded.min()) / (x1_decoded.max() - x1_decoded.min() + 1e-8)
        x_interp_norm = (x_interp - x_interp.min()) / (x_interp.max() - x_interp.min() + 1e-8)
        metrics_ssim.append(ssim(x1_norm, x_interp_norm, data_range=1.0))
        metrics_psnr.append(psnr(x1_norm, x_interp_norm, data_range=1.0))

    # Plot metrics curve in last column
    ax_metric = axes[sample_idx, -2]
    ax_metric.plot(alphas.cpu().numpy(), metrics_ssim, marker="o", label="SSIM")
    ax_metric.set_title("SSIM" if sample_idx == 0 else "")
    if sample_idx == 0:
        ax_metric.legend()

    ax_metric = axes[sample_idx, -1]
    ax_metric.plot(alphas.cpu().numpy(), metrics_psnr, marker="s", label="PSNR")
    ax_metric.set_title("PSNR" if sample_idx == 0 else "")
    if sample_idx == 0:
        ax_metric.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_root, "all_samples_results.png"))
plt.close(fig)

print(f"Saved combined results graph at {os.path.join(output_root, 'all_samples_results.png')}")
import os, gc
import torch
import torch.nn.functional as F
import pandas as pd
from torch.amp import autocast, GradScaler

from monai.utils import set_determinism
from monai.networks.schedulers import DDPMScheduler
from monai.inferers import DiffusionInferer
# from src.mask_diffusion_inferer import MaskDiffusionInferer
from tqdm import tqdm
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt

# from src.diffusion import (
#     sample_using_diffusion
# )
from accelerate import Accelerator
from PIL import Image

from src import utils_usage as utils  
from utils import args, import_from_dotted_path, utils_metric

from collections import OrderedDict
from copy import deepcopy

from src import diffusion
from src.ct_2D_latent_dataloader import create_paired_dataloader
from accelerate import DistributedDataParallelKwargs

from src import networks
from src import init_autoencoder

# from monai.generative.losses import PerceptualLoss
from src.flow import compute_ut, compute_xt

# from step1.model2D import utils_usage as utils

from accelerate.utils import DistributedDataParallelKwargs

# Set the desired behavior
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

# Initialize accelerator with custom DDP config
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

DEVICE = accelerator.device

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
set_determinism(0)


os.makedirs(args.cache_dir,  exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)

from torch import nn
from monai.networks.schedulers import DDIMScheduler
import imageio.v2 as imageio
from monai.networks.schedulers.ddpm import DDPMPredictionType
from monai.networks.schedulers.ddim import DDIMPredictionType




import torch


class EMA:
    def __init__(self, model, beta=0.9999, update_after_step=100, update_every=1):
        """
        model: torch.nn.Module
        beta: decay rate (closer to 1 → slower update, smoother EMA)
        update_after_step: start EMA updates only after this many steps
        update_every: update EMA every N steps
        """
        self.beta = beta
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.step = 0

        # Create a copy of the model for EMA
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

    def update(self, model):
        self.step += 1
        if self.step < self.update_after_step:
            # keep ema weights the same before warmup
            self._copy_params(model)
            return
        if self.step % self.update_every != 0:
            return

        with torch.no_grad():
            msd = model.state_dict()
            msd = remove_module_prefix(model.state_dict())

            for k, ema_v in self.ema_model.state_dict().items():
                model_v = msd[k].detach()
                if not model_v.dtype.is_floating_point:
                    ema_v.copy_(model_v)  # buffers (int, bool etc.)
                else:
                    ema_v.mul_(self.beta).add_(model_v, alpha=1 - self.beta)

    def _copy_params(self, model):
        """copy params from model → ema_model (for init/warmup)"""
        w = remove_module_prefix(model.state_dict())
        self.ema_model.load_state_dict(w)

    def state_dict(self):
        return remove_module_prefix(self.ema_model.state_dict())

    def to(self, device):
        self.ema_model.to(device)
        return self




class TimeScheduler:
    def __init__(self, device=DEVICE, sigma_min=0.05, sigma_max=1.0):
        """
        device: torch device
        sigma_min: minimum noise scale
        sigma_max: maximum noise scale (for inference schedule)
        """
        self.device = device
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    # ---------------- TRAINING ---------------- #
    def sample_train_t(self, batch_size, method="uniform", eps=1e-3):
        """
        Sample timesteps for training.
        method: "uniform" or "beta"
        eps: avoid exactly 0 or 1
        """
        if method == "uniform":
            t = torch.rand(batch_size, device=self.device)
        elif method == "beta":
            t = torch.distributions.Beta(2.0, 2.0).sample((batch_size,)).to(self.device)
        else:
            raise ValueError(f"Unknown training sampling method: {method}")
        return torch.clamp(t, eps, 1 - eps)

    # ---------------- INFERENCE ---------------- #
    def sigma_of_t(self, t):
        """σ(t) = σ_min * sqrt(t * (1 - t))"""
        return self.sigma_min * torch.sqrt(t * (1 - t))

    def invert_sigma(self, sigma, num_points=10000):
        """
        Numerically invert σ(t) → t by lookup.
        """
        t_grid = torch.linspace(0, 1, num_points, device=self.device)
        sigma_grid = self.sigma_of_t(t_grid)
        idx = torch.argmin((sigma_grid[:, None] - sigma[None, :]).abs(), dim=0)
        return t_grid[idx]

    def make_inference_t_steps(self, num_steps):
        """
        Make inference timesteps using log-uniform σ schedule.
        """
        sigmas = torch.exp(
            torch.linspace(torch.log(torch.tensor(self.sigma_max, device=self.device)),
                           torch.log(torch.tensor(self.sigma_min, device=self.device)),
                           num_steps, device=self.device)
        )
        t_steps = self.invert_sigma(sigmas)
        return t_steps



t_schedule = TimeScheduler(device=DEVICE, sigma_min=0.05, sigma_max=1.0)


@torch.no_grad()
def sample_using_diffusion(
        autoencoder: nn.Module,
        diffusion: nn.Module,
        x0, x1,
        device: str,
        scale_factor: int = 1,
        num_training_steps: int = 1000,
        num_inference_steps: int = 20,
        schedule: str = 'scaled_linear_beta',
        beta_start: float = 0.0015,
        beta_end: float = 0.0205,
        verbose: bool = True,
        epoch: int = 0,                   # save plot per epoch
        save_dir: str = "error_plot"      # save directory
) -> torch.Tensor:
    """
    Sampling random brain MRIs that follow the covariates in `context`.
    Tracks MAE vs timestep and saves as a plot.
    """

    # initialize latent with x0
    z = x0.clone()

    # Linear
    #
    t_steps = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device)
    # dt = t_steps[1] - t_steps[0]

    # sigma_max = 1.0    # largest noise (you can tune)
    # sigma_min = 0.001  # smallest noise

    # # log-uniform interpolation of σ
    # sigmas = torch.exp(
    #     torch.linspace(torch.log(torch.tensor(sigma_max)),
    #                 torch.log(torch.tensor(sigma_min)),
    #                 num_inference_steps+1, device=device)
    # )
    # t_steps = invert_sigma(sigmas, sigma_min=0.05, device=device)

    # Beta(2,2)

    # t_steps = t_schedule.make_inference_t_steps(num_inference_steps + 1)


    mae_values = []  # store MAE per timestep
    mae_values = []
    progress_bar = tqdm(range(num_inference_steps), desc="Sampling", disable=not verbose)

    for i in progress_bar:
        dt = t_steps[i+1] - t_steps[i]

        timestep = t_steps[i].expand(z.shape[0])  # match batch size
        v = diffusion(z.float(), timestep)

        z = z + v * dt  # Euler update

        mae = torch.mean(torch.abs(z - x1)).item()
        mae_values.append(mae)

    # final decoded output
    z = z / scale_factor[1]
    x = autoencoder.decode(z.to(device)).sample.cpu()

    # -------------------
    # Plot MAE vs timestep
    # -------------------
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"img_epoch{epoch:04d}.png")

    plt.figure(figsize=(7, 5))
    plt.plot(range(num_inference_steps), mae_values, marker="o")
    plt.xlabel("Diffusion Timestep")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title(f"MAE vs. Diffusion Timestep (Epoch {epoch})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(
        "MAE Stats:",
        "start =", mae_values[0],
        "end =", mae_values[-1],
        "min =", min(mae_values),
        "max =", max(mae_values)
    )


    return x

mask_key       = "mask"
latent_key     = "CTC"
file_key       = "CT_path"  # CTC_path
broken_latent_key = "CT"


def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = v
    return new_state_dict


def save_image(path, x):
    """
    Save a single image to the specified path.
    """

    x = np.clip(x, 0, 1)
    x = (x * 255).astype(np.uint8)

    # Save the image
    imageio.imwrite(path, x)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def save_image(path, image_array):
    """Convert np array to uint8 image and save"""
    image_array = np.clip(image_array, 0, 1)  # normalize for safety
    image_array = (image_array * 255).astype(np.uint8)
    Image.fromarray(image_array).save(path)

def get_middle_slices(volume):  # [C, D, H, W]
    d, h, w = volume.shape[1:]
    axial = volume[:, d // 2, :, :]     # shape: [C, H, W]
    coronal = volume[:, :, h // 2, :]   # shape: [C, D, W]
    sagittal = volume[:, :, :, w // 2]  # shape: [C, D, H]
    return [axial, coronal, sagittal]

def save_grid_image_by_plane(image_np, recon_broken, recon_np, save_root, b, epoch, modality_names=None):
    image_channels = image_np[b].shape[0]
    row_labels = modality_names[:image_channels] if modality_names else [f"Mod{i}" for i in range(image_channels)]
    row_labels += ["Mask"]
    col_labels = ["Input", "Broken", "Recon"]
    total_rows = image_channels + 1
    total_cols = 3  # input, broken, recon
    plane_names = ["Axial", "Coronal", "Sagittal"]

    image_views  = get_middle_slices(image_np[b])
    broken_views = get_middle_slices(recon_broken[b])
    recon_views  = get_middle_slices(recon_np[b])


    for p, plane in enumerate(plane_names):
        fig, axes = plt.subplots(total_rows, total_cols, figsize=(total_cols * 2, total_rows * 2))

        for r in range(image_channels):
            # Input, Broken, Recon for modality r
            axes[r, 0].imshow(image_views[p][r], cmap="gray")
            axes[r, 1].imshow(broken_views[p][r], cmap="gray")
            axes[r, 2].imshow(recon_views[p][r], cmap="gray")
            for c in range(total_cols):
                axes[r, c].axis("off")
            axes[r, 0].set_ylabel(row_labels[r], fontsize=12)

        # Last row: only show mask
        axes[image_channels, 1].axis("off")
        axes[image_channels, 2].axis("off")
        axes[image_channels, 0].axis("off")
        axes[image_channels, 0].set_ylabel("Mask", fontsize=12)

        # Column headers
        for c in range(total_cols):
            axes[0, c].set_title(col_labels[c], fontsize=12)

        fig.suptitle(f"Sample {b} - {plane} View", fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(save_root, f"{epoch}_sample{b}_{plane}.jpg")
        plt.savefig(save_path, dpi=150)
        plt.close()

import imageio

def save_(image, save_root, epoch=0, modality_names=None, fname="stacked.png"):
    """
    Save a stacked numpy image as PNG.

    Args:
        image (np.ndarray): 2D or 3D array.
            If 2D: (H, W)
            If 3D: (H, W, C)
        save_root (str): directory to save file
        epoch (int): epoch number for filename
        modality_names (list[str]): optional names for modalities
        fname (str): custom filename
    """
    os.makedirs(save_root, exist_ok=True)
    out_path = os.path.join(save_root, f"epoch{epoch:03d}_{fname}")

    # normalize if needed
    if image.dtype != np.uint8:
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:  # avoid div by zero
            image = (255 * (image - img_min) / (img_max - img_min)).astype(np.uint8)
        else:
            image = (image * 255).astype(np.uint8)

    # save
    imageio.imwrite(out_path, image)
    print(f"✅ Validation Saved: {out_path}")


from pathlib import Path
def stack_and_save(image_np, recon_broken, recon_np, save_root, epoch, modality_names):
    """
    Save a grid of [original, broken, recon] for each modality, with labels.

    Args:
        image_np, recon_broken, recon_np: np.ndarray (B, C, H, W)
        save_root (str | Path): output directory
        epoch (int): current epoch
        modality_names (list[str]): names for modalities (len = C)
    """
    B, C, H, W = image_np.shape
    ncols = 3 * C                # Original, Broken, Recon for each modality
    nrows = min(B, 3)             # up to 3 samples

    fig, axs = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axs = np.atleast_2d(axs)

    # set column headers
    for c, modality in enumerate(modality_names[:C]):
        axs[0, 3*c].set_title(f"{modality} - Original")
        axs[0, 3*c+1].set_title(f"{modality} - Broken")
        axs[0, 3*c+2].set_title(f"{modality} - Recon")

    for b in range(nrows):
        for c in range(C):
            axs[b, 3*c].imshow(image_np[b, c], cmap="gray", vmin=0, vmax=1)
            axs[b, 3*c+1].imshow(recon_broken[b, c], cmap="gray", vmin=0, vmax=1)
            axs[b, 3*c+2].imshow(recon_np[b, c], cmap="gray", vmin=0, vmax=1)

            # remove axes
            axs[b, 3*c].axis("off")
            axs[b, 3*c+1].axis("off")
            axs[b, 3*c+2].axis("off")

    plt.tight_layout()
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    save_path = save_root / f"stacked_epoch{epoch}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved grid with labels to", save_path)




use_standard_norm = True 

import torch

def standard_normalize(x, mean=None, std=None, eps=1e-8):
    """
    Standard-normalize a batch of images/volumes.
    
    Args:
        x   : torch.Tensor of shape (B, C, H, W) or (B, C, D, H, W)
        mean: torch.Tensor or None. If None, compute per-batch mean.
        std : torch.Tensor or None. If None, compute per-batch std.
        eps : small constant to avoid division by zero.
    
    Returns:
        x_norm: normalized tensor
        mean  : mean used for normalization
        std   : std used for normalization
    """
    if mean is None:
        mean = x.mean(dim=tuple(range(1, x.ndim)), keepdim=True)
    if std is None:
        std = x.std(dim=tuple(range(1, x.ndim)), keepdim=True) + eps

    x_norm = (x - mean) / std

    x_norm = torch.clamp(x_norm, -6, 6)

    return x_norm, mean, std


def denormalize(x_norm, mean, std):
    """
    Invert standard normalization.
    
    Args:
        x_norm: normalized tensor
        mean  : mean used in normalization
        std   : std used in normalization
    
    Returns:
        x: denormalized tensor
    """
    return x_norm * std + mean



def images_to_tensorboard(
        batch,
        writer,
        epoch,
        mode,
        autoencoder,
        diffusion,
        scale_factor,
        modality_names = ["T1c", "T1n", "T2w", "T2f"] 
):
    """
    Visualize the generation on tensorboard
    """

    ct_img = batch["CT_img"]
    ctc_img = batch["CTC_img"]

    if use_standard_norm:
        ct_img, ct_mean, ct_std    = standard_normalize(ct_img)
        ctc_img, ctc_mean, ctc_std = standard_normalize(ctc_img)

    x0 = batch[broken_latent_key].to(DEVICE).clone() * scale_factor[0]
    x1 = batch[latent_key].to(DEVICE).clone()        * scale_factor[1]

    ae = autoencoder  # .module if hasattr(autoencoder, "module") else autoencoder

    with torch.no_grad(), accelerator.autocast():
        image = sample_using_diffusion(
            autoencoder=autoencoder,
            diffusion=diffusion,
            x0=x0, #inputs_latents,
            x1=x1,
            num_inference_steps=200,
            device=DEVICE,
            scale_factor=scale_factor,
            epoch=epoch
        )

        recon_broken = ae.decode(x0 / scale_factor[0]).sample.cpu().numpy()  # [B, 1, H, W]
        recon_target = ae.decode(x1 / scale_factor[1]).sample.cpu().numpy()  # [B, 1, H, W]



    image_np     = recon_target  #.cpu().numpy()  # [B, 1, H, W]
    recon_broken = recon_broken  #.cpu().numpy()  # [B, 1, H, W]
    recon_np     = image.cpu().numpy()  #.max(axis=1, keepdims=True)  # [B, 3, H, W] -> [B, 1, H, W]



    save_root = "./fm_samples"
    os.makedirs(save_root, exist_ok=True)

   
    modality_names = [m.upper() for m in modality_names]

    if use_standard_norm:
        ctc_mean = ctc_mean.cpu().numpy()
        ctc_std  = ctc_std.cpu().numpy()
        recon_np     = denormalize(recon_np, ctc_mean, ctc_std) #.cpu().numpy()
        recon_broken = denormalize(recon_broken, ctc_mean, ctc_std)#.cpu().numpy()
        image_np     = denormalize(image_np, ctc_mean, ctc_std)#.cpu().numpy

    # print("image_np[b] =", image_np[b].shape, recon_broken[b].shape, recon_np[b].shape, context_np[b].shape)
    stack_and_save(image_np, recon_broken, recon_np, save_root, epoch, modality_names)
  
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

   
    def evaluate_images_modalities(image_np, recon_broken, recon_np, modality_names=None, idx=0):
        """
        Compute PSNR and SSIM per modality:
            - recon_np (Fake) vs image_np (Target)
            - recon_broken (Input) vs image_np (Target)

        Args:
            image_np: np.ndarray (B,C,H,W)
            recon_broken: np.ndarray (B,C,H,W)
            recon_np: np.ndarray (B,C,H,W)
            modality_names: list of modality names (len=C)
            idx: validation index
        """
        B, C, H, W = image_np.shape
        if modality_names is None:
            modality_names = [f"Modality-{c}" for c in range(C)]

        # Storage
        psnr_after, ssim_after = [[] for _ in range(C)], [[] for _ in range(C)]
        psnr_before, ssim_before = [[] for _ in range(C)], [[] for _ in range(C)]

        for b in range(B):
            for c in range(C):
                target    = image_np[b, c]
                input_img = recon_broken[b, c]
                fake_img  = recon_np[b, c]

                rng = target.max() - target.min() if target.max() > target.min() else 1.0

                # Fake vs Target
                psnr_after[c].append(psnr(target,  fake_img, data_range=rng))
                ssim_after[c].append(ssim(target,  fake_img, data_range=rng))

                # Input vs Target
                psnr_before[c].append(psnr(target, input_img, data_range=rng))
                ssim_before[c].append(ssim(target, input_img, data_range=rng))


        # Inout
        ctc_fake_vs_ct_true_psnr, ctc_fake_vs_ct_true_ssim = [], []
        if C >= 2:
            for b in range(B):
                target = image_np[b, 0]   # CT True
                fake   = recon_np[b, 1]   # CTC Fake
                rng = target.max() - target.min() if target.max() > target.min() else 1.0
                ctc_fake_vs_ct_true_psnr.append(psnr(target, fake, data_range=rng))
                ctc_fake_vs_ct_true_ssim.append(ssim(target, fake, data_range=rng))
            

        # Print nicely
        print(f"\n[Validation Summary over Epochs {idx}]")
        print("----------------------------------------------------------------------------------------------------")
        print(f"{'Modality':15} | {'Fake→Target PSNR':18} | {'Input→Target PSNR':18} | {'Fake→Target SSIM':18} |  {'Input→Target SSIM':18}")
        print("----------------------------------------------------------------------------------------------------")
        for c, name in enumerate(modality_names):
            print(f"{name:15} | {np.mean(psnr_after[c]):.3f}{'':12} | {np.mean(psnr_before[c]):.3f}{'':12} | {np.mean(ssim_after[c]):.3f}{'':12} |  {np.mean(ssim_before[c]):.3f}{'':12}")
        
        print("----------------------------------------------------------------------------------------------------")

       
        print(f"\n[Cross-Modality Comparison]")
        print("----------------------------------------------------------------------------------------------------")
        print(f"{'Comparison':15} | {'Fake→CT True PSNR':18} | {'Fake→CT True SSIM':18}")
        print("----------------------------------------------------------------------------------------------------")
        print(f"{'CTC Fake vs CT':15} | {np.mean(ctc_fake_vs_ct_true_psnr):.3f}{'':12} | {np.mean(ctc_fake_vs_ct_true_ssim):.3f}{'':12}")
        print("----------------------------------------------------------------------------------------------------")


    evaluate_images_modalities(image_np, recon_broken, recon_np, modality_names=modality_names, idx=epoch)




# from src.MM_AE import AutoencoderKL_multi_encoder

from SM_AE import AutoencoderKL_single_encoder

def define_2DAE(in_channels=3,  out_channels=2, latent_channels = 16):
    from huggingface_hub import snapshot_download
    from diffusers import AutoencoderKL

    
    from diffusers.models import AutoencoderKL
    # block_out_channels=(256, 512)  # (256, 512)
    block_out_channels=(128, 256)  # (256, 512)
    layers_per_block = 3

    # block_out_channels=(128, 128, 256) # 128, 256
    # layers_per_block = 2

    n_blocks = len(block_out_channels)

    vae = AutoencoderKL_single_encoder(
        num_encode = in_channels,
        in_channels=1, out_channels=out_channels, latent_channels=latent_channels,
        block_out_channels=block_out_channels, 
        down_block_types=("DownEncoderBlock2D",) * n_blocks,
        up_block_types=("UpDecoderBlock2D",) * n_blocks,
        layers_per_block=layers_per_block,   # 3
        norm_num_groups=32
    )



    return vae


def charbonnier_smooth_l1_loss(pred, target, beta=0.05, eps=1e-6, reduction='mean'):
    """
    Charbonnier-SmoothL1 hybrid loss:
    sqrt( (SmoothL1(pred, target))^2 + eps^2 )

    Args:
        pred (Tensor): predictions
        target (Tensor): ground truth
        beta (float): SmoothL1 transition point
        eps (float): stability term for Charbonnier
        reduction (str): 'mean', 'sum', or 'none'
    """
    # elementwise SmoothL1 (no reduction yet)
    diff = F.smooth_l1_loss(pred, target, beta=beta, reduction='none')
    # wrap with Charbonnier
    loss = torch.sqrt(diff * diff + eps * eps)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
                        
if __name__ == '__main__':
    image_key = args.input_modality

    if isinstance(image_key, (list, tuple)):
        image_key_str = "-".join(image_key)
    else:
        image_key_str = str(image_key)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------- Define Dataloader ----------------
    num_train_timesteps = 1000
    # (16, 128, 128) , latent

    dimension = 2
    key_to_load  = []         
    key_to_load.extend(image_key)


    data_root    = args.data_dir
    latent_root  = args.latent_dir  # wherever you want latents
    spatial_size = [256, 256]  # 512 or 256

    target_dataset = ["Adrenal"]
    # target_dataset = ["Adrenal", "Bladder", "Lung", "Stomach", "Uterus"]

    image_key_str +=  "-" + "-".join(target_dataset)


    train_loader, train_ds     = create_paired_dataloader(csv_path=args.dataset_csv, target_dataset=target_dataset, 
                                                          data_root=data_root, out_root=latent_root,
                                                          split="train", batch_size=args.batch_size, spatial_size=spatial_size)
    test_loader,  test_ds      = create_paired_dataloader(csv_path=args.dataset_csv, target_dataset=target_dataset, 
                                                          data_root=data_root, out_root=latent_root,
                                                          split="test",  batch_size=4 * args.batch_size, spatial_size=spatial_size)


    in_channels = len(image_key)
    print("Setting up Autoencoder model...")
    autoencoder   = define_2DAE(in_channels=in_channels, 
                                out_channels=in_channels,
                                latent_channels = 16).to(DEVICE).float()
    
    print("Finish setting up...")
    try:
        weight = torch.load(args.aekl_ckpt)
        weight = remove_module_prefix(weight)

        autoencoder.load_state_dict(weight)

    except FileNotFoundError:
        print(f"File {args.aekl_ckpt} not found, using random initialization for autoencoder.")
        
    autoencoder.to(DEVICE)
    autoencoder.eval()  # Important for inference

    # from generative.networks.nets import DiffusionModelUNet
    # in_channels=16
    # model = DiffusionModelUNet(
    #     spatial_dims=2,  # 2D
    #     in_channels=in_channels,  # x
    #     out_channels=in_channels,  # predice delta_x_t
    # )


    diffusion = networks.init_latent_diffusion(args, in_channels=16, use_image=False, spatial_dims=dimension).to(DEVICE)


    ema_diffusion = EMA(diffusion)


    if args.diff_ckpt:
        print("Loading diffusion checkpoint:", args.diff_ckpt)
        weight = torch.load(args.diff_ckpt, map_location=DEVICE)
        weight = remove_module_prefix(weight)
        diffusion.load_state_dict(weight, strict=True)
        print("=> Loaded Diffusion successfully.")
    else:
        print(f"=> No diffusion checkpoint provided, using random init for diffusion model.")

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr, weight_decay=1e-6)  # AdamW

    scheduler_g = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=args.lr * 0.1,      # minimum LR
        max_lr =args.lr,             # maximum LR
        step_size_up=5000,          # number of steps to go from base_lr → max_lr
        mode="triangular2",         # triangular2 decays amplitude after each cycle
        cycle_momentum=False        # must be False for AdamW
    )

    a = train_loader.dataset[0][latent_key]
    rate_num = 25

    scale_factor = []
    with torch.no_grad(), accelerator.autocast():
        z_list = [train_loader.dataset[i][latent_key] for i in range(rate_num)]
        z = torch.stack(z_list, dim=0)  # Stack into a single tensor
        scale_factor_full = 1 / torch.std(z)  

        z_list = [train_loader.dataset[i][broken_latent_key] for i in range(rate_num)]
        z = torch.stack(z_list, dim=0)  # Stack into a single tensor
        scale_factor_broken = 1 / torch.std(z)  
        scale_factor = [scale_factor_broken, scale_factor_full]

    print(f"Scaling factor set to {scale_factor}")

    from monai.losses import PerceptualLoss
    use_adv           = True  # NAN
    adv_weight        = 0.025
    perceptual_weight = 0.1     # if  args.use_broken else 0.0  # 0.1  
    kl_weight         = 1e-7  # 1e-7
    perc_loss_fn = PerceptualLoss(spatial_dims=dimension,
                                      network_type="squeeze",
                                      is_fake_3d=True if dimension == 3 else False).to(DEVICE)

    from monai.losses import PerceptualLoss, PatchAdversarialLoss
    from src import init_patch_discriminator # KLDivergenceLoss

    # kl_loss_fn  = KLDivergenceLoss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")  # criterion="hinge"

    discriminator = init_patch_discriminator(args.disc_ckpt, 
                                             spatial_dims=dimension,
                                             in_channels=16, 
                                             num_layers_d=3).to(DEVICE)
 

    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr * 0.1, weight_decay=1e-6)  # AdamW

    

    autoencoder, optimizer, train_loader, diffusion, scheduler_g, autoencoder.encode, autoencoder.decode, perc_loss_fn, ema_diffusion, adv_loss_fn, discriminator = accelerator.prepare(
        autoencoder, optimizer, train_loader, diffusion, scheduler_g, autoencoder.encode, autoencoder.decode, perc_loss_fn, ema_diffusion, adv_loss_fn, discriminator
    )

    # writer   = SummaryWriter()
    global_counter = {'train': 0}  # , 'valid': 0 }
    loaders  = {'train': train_loader}  # , # 'valid': valid_loader }
    datasets = {'train': train_loader.dataset}  # , 'valid': validset }

    ae = autoencoder #.module if hasattr(autoencoder, "module") else autoencoder
    gradient_accumulation_steps = args.grad_accum_steps if hasattr(args, 'grad_accum_steps') else 4  # for example
    
    for p in ae.parameters():
        p.requires_grad = False
    
    ae.eval()

    for epoch in range(args.n_epochs):

        for mode in loaders.keys():
            loader = loaders[mode]
            diffusion.train() if mode == 'train' else diffusion.eval()
            epoch_loss = 0
            mse_loss_total = 0.0
            rec_loss_total = 0.0
            gen_loss_total = 0.0
            ae_loss_total  = 0.0
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"{mode.upper()} Epoch {epoch}")

            for step, batch in progress_bar:
                if step > 500:
                    break

                if args.DEBUG and step >= 10:
                    print(f"[DEBUG] Step {step}: {batch[latent_key].shape}")
                    break

                # with autocast(device_type='cuda',enabled=True):
                with accelerator.autocast():
                    if mode == 'train': optimizer.zero_grad(set_to_none=True)

                    # Use to be  context: context tensor (N, 1, ContextDim).
                    B         = batch[latent_key].shape[0]
                    t         = torch.rand(B, device=DEVICE)  # log-uniform?
                    # t = torch.sigmoid(torch.randn(B, device=DEVICE))
                    
                    # t = t_schedule.sample_train_t(batch_size=B, method="beta")

                    x0        = batch[broken_latent_key].to(DEVICE).clone() * scale_factor[0]
                    x1        = batch[latent_key].to(DEVICE).clone() * scale_factor[1]
                    ctc_image = batch["CTC_img"].to(DEVICE)

                    ctc_norm, ctc_image_mean, ctc_image_std = standard_normalize(ctc_image)


                    # x0 -> x1
                    sigma = 0.05 # 1.0
                    xt, eps, sigma_t = compute_xt(x0, x1, t, sigma_min=sigma)
                    ut = compute_ut(x0, x1, t, eps, sigma_t, sigma_min=sigma)

                    # x_t = (1 - t_img) * x_0 + t_img * x_1         # [B, 1, H, W]
                    # ut = x_1 - x_0                              # [B, 1, H, W]
                   
                    pred = diffusion(xt, t) #x=xt, timesteps=t, context=None)

                    mse_loss = F.mse_loss(pred, ut)  # MSE Loss
                    x1_pred = x0 + pred

                    import torch
                    def flatten_latent(z):
                        """
                        z: latent tensor of shape (B, C, H, W) or (B, C, H, W, D)
                        returns: (B, C*H*W*D)
                        """
                        return z.view(z.size(0), -1)


                    def compute_mmd(x, y, sigma=1.0):
                        """
                        Compute MMD between two sets of latents
                        x: (B, C, H, W, ...)  -> will be flattened
                        y: (B, C, H, W, ...)  -> will be flattened
                        """
                        x = flatten_latent(x)
                        y = flatten_latent(y)

                        xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
                        
                        rx = xx.diag().unsqueeze(0).expand_as(xx)
                        ry = yy.diag().unsqueeze(0).expand_as(yy)

                        dxx = rx.t() + rx - 2*xx
                        dyy = ry.t() + ry - 2*yy
                        dxy = rx.t() + ry - 2*xy

                        kxx = torch.exp(-dxx / (2*sigma**2))
                        kyy = torch.exp(-dyy / (2*sigma**2))
                        kxy = torch.exp(-dxy / (2*sigma**2))

                        return kxx.mean() + kyy.mean() - 2*kxy.mean()



                    rec_loss = 0.1 * compute_mmd(x1_pred, x1)
                    # loss_mmd = compute_mmd(z1_to_2, z2)

                    loss = mse_loss + rec_loss
                    
                    # adv_loss_fn
                if use_adv:
                    logits_fake = discriminator(x1_pred.contiguous())[-1]
                    gen_loss = adv_weight * adv_loss_fn(logits_fake, 
                                                        target_is_real=True, 
                                                        for_discriminator=False)
                    loss += gen_loss
                else:
                    gen_loss = torch.tensor(0.0)

                   

                recon = ae.decode((x1_pred / scale_factor[1])).sample
                recon = recon[:, 1:2, :, :] #.unsqueeze(1)  # only CTC channel
                ae_loss = F.l1_loss(recon, ctc_norm)  # 0.1
                loss += ae_loss

                    # kld_loss = kl_weight * ( kl_loss_fn(z_mu, z_sigma) )

                #     if epoch > 10:
                #         
                        

                # if epoch > 10:
                #     reconstruction_3ch = torch.cat([recon] * 3, dim=1)  # [B, C, H, W]
                #     images_3ch         = torch.cat([ctc_norm] * 3, dim=1)
                    
                    
                #     loss += 0.1 *  perc_loss_fn(
                #         reconstruction_3ch.float(),
                #         images_3ch.float().detach()
                #     )


                    # cos_loss = 1 - F.cosine_similarity(pred, ut, dim=1).mean()
                    # mag_loss = charbonnier_smooth_l1_loss(pred, ut)
                    # loss = cos_loss + 0.1 * mag_loss



                if mode == 'train':
                    # Accumulated Loss
                    loss_acc = loss / gradient_accumulation_steps  # normalize loss
                    accelerator.backward(loss_acc)

                    if (step + 1) % gradient_accumulation_steps == 0 or (step + 1 == len(loader)):
                        optimizer.step()
                        scheduler_g.step() 
                        ema_diffusion.update(diffusion)  

                epoch_loss += loss.item()
                mse_loss_total += mse_loss.item()
                rec_loss_total += rec_loss.item()
                gen_loss_total += gen_loss.item()
                ae_loss_total  += ae_loss.item()

                progress_bar.set_postfix({
                    "Step": step,
                    "Loss": epoch_loss / (step + 1),
                    "MSE":  mse_loss_total / (step + 1),
                    "Rec":  rec_loss_total / (step + 1),
                    "Gen":  gen_loss_total / (step + 1),
                    "AE":   ae_loss_total / (step + 1),
                    # "Percept": perceptual_loss.item(),
                })

                global_counter[mode] += 1


                 # ADV Loss
                if use_adv:
                    optimizer_d.zero_grad()
                    # with accelerator.autocast():
                    fake_images = x1_pred.detach()  # Detach to cut generator graph
                    logits_real = discriminator(x1.contiguous())[-1]   # .contiguous().detach()
                    d_loss_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)

                    discriminator_loss = (d_loss_real) * 0.5
                    loss_d = discriminator_loss

                    optimizer_d.zero_grad()
                    accelerator.backward(loss_d)

                    del logits_real, loss_d
                
                    # with accelerator.autocast():
                    fake_images = x1_pred.detach()  # Detach to cut generator graph
                    logits_fake = discriminator(fake_images.contiguous())[-1]

                    d_loss_fake = adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)

                    discriminator_loss = (d_loss_fake) * 0.5
                    loss_d1 = discriminator_loss

                    accelerator.backward(loss_d1)
                    optimizer_d.step()
                    optimizer_d.zero_grad()


            # end of epoch
            epoch_loss = epoch_loss / len(loader)
            # writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch)


            accelerator.wait_for_everyone()
            # visualize results
            
            if accelerator.is_main_process:
                # ema_diffusion.apply_shadow() 
                images_to_tensorboard(
                    batch=batch,
                    writer=None,
                    epoch=epoch,
                    mode=mode,
                    autoencoder=autoencoder,
                    diffusion=ema_diffusion.ema_model if  epoch % 2 == 0 else diffusion,
                    scale_factor=scale_factor,
                    modality_names=image_key
                )
                # ema_diffusion.restore()

        # save the model                
        savepath = os.path.join(args.output_dir, f'dm-unet-ep-{epoch}-{image_key_str}.pth')
        # torch.save(diffusion.state_dict(), savepath)
        

        if accelerator.is_main_process:
            accelerator.save(diffusion.state_dict(), savepath)
            try:
                savepath = os.path.join(args.output_dir, f'dm-unet-ep-{epoch - 1}-{image_key_str}.pth')
                os.remove(savepath)
            except FileNotFoundError:
                print(f"File {savepath} not found, skipping deletion.")

        print("Saving models to: ", savepath)

        gc.collect()
        torch.cuda.empty_cache()

        accelerator.wait_for_everyone()

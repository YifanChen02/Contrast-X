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



@torch.no_grad()
def sample_using_diffusion(
        autoencoder: nn.Module,
        diffusion: nn.Module,
        x0, x1,
        device: str,
        scale_factor: int = 1,
        num_training_steps: int = 1000,
        num_inference_steps: int = 50,
        schedule: str = 'scaled_linear_beta',
        beta_start: float = 0.0015,
        beta_end: float = 0.0205,
        verbose: bool = True
) -> torch.Tensor:
    """
    Sampling random brain MRIs that follow the covariates in `context`.

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet
        context (torch.Tensor): the covariates
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): the scale factor (see Rombach et Al, 2021). Defaults to 1.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.
    Returns:
        torch.Tensor: the inferred follow-up MRI
    """


    # x0, x1,
    z = x0
    t_steps = torch.linspace(0.0, 1.0, num_inference_steps, device=device)


    progress_bar = tqdm(range(len(t_steps)), desc="Sampling", disable=not verbose)

    dt = t_steps[1] - t_steps[0] 


    for i, t in enumerate(progress_bar):
        with torch.no_grad(), accelerator.autocast():
            timestep = torch.tensor([t_steps[i]]).to(device)
            
            v = diffusion(x=z.float(), timesteps=timestep,)  # [B, C, ...] = v(x_t, t)

            z = z + v * dt

    # decode the latent
    z = z / scale_factor
    # z = utils.to_vae_latent_trick(z.cpu(), unpadded_z_shape=(z.shape[0], 4, *[s // 2 for s in spatial_size]))
    x = autoencoder.decode(z.to(device)).sample.cpu()  #.sample.cpu().squeeze(1)

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


def stack_and_save(image_np, recon_broken, recon_np, save_root, epoch, modality_names):
    """
    Stack [original, broken, recon] images for visualization.

    Args:
        image_np, recon_broken, recon_np: np.ndarray (B, C, H, W)
        save_root (str): output directory
        epoch (int): current epoch
        modality_names (list[str]): names for modalities
    """
    image_stacked = []
    for b in range(min(image_np.shape[0], 3)):  # only save up to 3 samples
        image_c = []
        for c in range(image_np.shape[1]):
            # horizontally concat triplet
            image_c.append(image_np[b, c])
            image_c.append(recon_broken[b, c])
            image_c.append(recon_np[b, c])

        # stack modalities vertically → (C*H, 3*W)
        image_c = np.concatenate(image_c, axis=-1)
        image_stacked.append(image_c)

    # stack batch samples vertically → (B*C*H, 3*W)
    image_stacked = np.concatenate(image_stacked, axis=0)

    # save
    save_(image_stacked, save_root, epoch=epoch, fname="stacked.png",
        modality_names=modality_names[:image_np.shape[1]])


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

    x0 = batch[latent_key].to(DEVICE).clone() * scale_factor
    x1 = batch[broken_latent_key].to(DEVICE).clone() * scale_factor

    # print("inputs_latents = ", inputs_latents.shape, "context = ", context.shape)

    ae = autoencoder.module if hasattr(autoencoder, "module") else autoencoder

    with torch.no_grad(), accelerator.autocast():
        image = sample_using_diffusion(
            autoencoder=ae,
            diffusion=diffusion,
            x0=x0, #inputs_latents,
            x1=x1,
            num_inference_steps=100,
            device=DEVICE,
            scale_factor=scale_factor
        )

        recon_origin = ae.decode(x0 / scale_factor).sample.cpu().numpy()  # [B, 1, H, W]
        recon_broken = ae.decode(x1 / scale_factor).sample.cpu().numpy()  # [B, 1, H, W]


    image_np     = recon_origin  #.cpu().numpy()  # [B, 1, H, W]
    recon_np     = image.cpu().numpy()  #.max(axis=1, keepdims=True)  # [B, 3, H, W] -> [B, 1, H, W]

    save_root = "./fm_samples"
    os.makedirs(save_root, exist_ok=True)

   
    modality_names = [m.upper() for m in modality_names]

    if use_standard_norm:
        ct_mean = ct_mean.cpu().numpy()
        ct_std  = ct_std.cpu().numpy()
        recon_np = denormalize(recon_np, ct_mean, ct_std) #.cpu().numpy()
        recon_broken = denormalize(recon_broken, ct_mean, ct_std)#.cpu().numpy()
        image_np = denormalize(image_np, ct_mean, ct_std)#.cpu().numpy

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
                psnr_after[c].append(psnr(target, fake_img, data_range=rng))
                ssim_after[c].append(ssim(target, fake_img, data_range=rng))

                # Input vs Target
                psnr_before[c].append(psnr(target, input_img, data_range=rng))
                ssim_before[c].append(ssim(target, input_img, data_range=rng))

        # Print nicely
        print(f"\n[Validation Summary over Epochs {idx}]")
        print("----------------------------------------------------------------------------------------------------")
        print(f"{'Modality':15} | {'Fake→Target PSNR':18} | {'Input→Target PSNR':18} | {'Fake→Target SSIM':18} |  {'Input→Target SSIM':18}")
        print("----------------------------------------------------------------------------------------------------")
        for c, name in enumerate(modality_names):
            print(f"{name:15} | {np.mean(psnr_after[c]):.3f}{'':12} | {np.mean(psnr_before[c]):.3f}{'':12} | {np.mean(ssim_after[c]):.3f}{'':12} |  {np.mean(ssim_before[c]):.3f}{'':12}")
        print("----------------------------------------------------------------------------------------------------")



    evaluate_images_modalities(image_np, recon_broken, recon_np, modality_names=modality_names, idx=epoch)



def define_2DAE(in_channels=3,  out_channels=2, latent_channels = 16):
    from huggingface_hub import snapshot_download
    from diffusers import AutoencoderKL
    
    from diffusers.models import AutoencoderKL

    block_out_channels=(256, 512)
    layers_per_block = 3

    n_blocks = len(block_out_channels)

    vae = AutoencoderKL(
        in_channels=in_channels, out_channels=out_channels, latent_channels=latent_channels,
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

    train_loader, train_ds     = create_paired_dataloader(csv_path=args.dataset_csv, 
                                                          data_root=data_root, out_root=latent_root,
                                                          split="train", batch_size=args.batch_size, spatial_size=spatial_size)
    test_loader,  test_ds      = create_paired_dataloader(csv_path=args.dataset_csv, data_root=data_root, out_root=latent_root,
                                                          split="test",  batch_size=args.batch_size, spatial_size=spatial_size)


    in_channels = len(image_key)
    print("Setting up Autoencoder model...")
    autoencoder   = define_2DAE(in_channels=in_channels * 2, 
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


    diffusion = networks.init_latent_diffusion(args, in_channels=16, use_image=False, spatial_dims=dimension).to(DEVICE)
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

    with torch.no_grad(), accelerator.autocast():
        z_list = [train_loader.dataset[i][latent_key] for i in range(5)]
        z = torch.stack(z_list, dim=0)  # Stack into a single tensor


    scale_factor = 1 / torch.std(z)  
    print(f"Scaling factor set to {scale_factor}")

    autoencoder, optimizer, train_loader, diffusion, scheduler_g = accelerator.prepare(
        autoencoder, optimizer, train_loader, diffusion, scheduler_g
    )

    # writer   = SummaryWriter()
    global_counter = {'train': 0}  # , 'valid': 0 }
    loaders  = {'train': train_loader}  # , # 'valid': valid_loader }
    datasets = {'train': train_loader.dataset}  # , 'valid': validset }

    ae = autoencoder.module if hasattr(autoencoder, "module") else autoencoder
    gradient_accumulation_steps = args.grad_accum_steps if hasattr(args, 'grad_accum_steps') else 4  # for example


    for epoch in range(args.n_epochs):

        for mode in loaders.keys():
            loader = loaders[mode]
            diffusion.train() if mode == 'train' else diffusion.eval()
            epoch_loss = 0
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"{mode.upper()} Epoch {epoch}")

            for step, batch in progress_bar:
                if args.DEBUG and step >= 10:
                    print(f"[DEBUG] Step {step}: {batch[latent_key].shape}")
                    break

                # with autocast(device_type='cuda',enabled=True):
                with accelerator.autocast():
                    if mode == 'train': optimizer.zero_grad(set_to_none=True)

                    # Use to be  context: context tensor (N, 1, ContextDim).
                    B         = batch[latent_key].shape[0]
                    t         = torch.rand(B, device=DEVICE)
                    x0        = batch[latent_key].to(DEVICE).clone() * scale_factor
                    x1        = batch[broken_latent_key].to(DEVICE).clone() * scale_factor

                    # x0 -> x1
                    sigma = 0.01
                    xt = compute_xt(x0=x0, x1=x1, t=t, sigma_min=sigma)
                    ut = compute_ut(x0=x0, x1=x1, t=t)
                    
                    pred = diffusion(x=xt, timesteps=t, context=None)

                    

                    # loss = F.mse_loss(pred, ut)  # MSE Loss
                    loss = charbonnier_smooth_l1_loss(pred, ut)


                    # cos_loss = 1 - F.cosine_similarity(pred, ut, dim=1).mean()
                    # mag_loss = charbonnier_smooth_l1_loss(pred, ut)
                    # loss = cos_loss + 0.1 * mag_loss



                if mode == 'train':
                    # Accumulated Loss
                    loss = loss / gradient_accumulation_steps  # normalize loss
                    accelerator.backward(loss)

                    if (step + 1) % gradient_accumulation_steps == 0 or (step + 1 == len(loader)):
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler_g.step() 


                epoch_loss += loss.item()

                progress_bar.set_postfix({
                    "Step": step,
                    "Loss": epoch_loss / (step + 1),
                    # "Percept": perceptual_loss.item(),
                })

                global_counter[mode] += 1

            # end of epoch
            epoch_loss = epoch_loss / len(loader)
            # writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch)

            # visualize results
            images_to_tensorboard(
                batch=batch,
                writer=None,
                epoch=epoch,
                mode=mode,
                autoencoder=autoencoder,
                diffusion=diffusion,
                scale_factor=scale_factor,
                modality_names=image_key
            )



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

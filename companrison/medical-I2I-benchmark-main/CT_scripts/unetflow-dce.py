import os
import argparse

parser = argparse.ArgumentParser(description="Train a flow matching model from CT to CTC.")
parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
parser.add_argument('--lr_min', type=float, default=1e-6, help="Minimum learning rate for CosineAnnealingLR")
parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for DataLoader")
parser.add_argument('--batch_size', type=int, default=6, help="Batch size")
parser.add_argument('--n_epochs', type=int, default=300, help="Number of training epochs")


parser.add_argument('--dataset_csv', type=str, required=False, default=None,
                    help='Path to the csv file that contains the dataset information.')
parser.add_argument('--data_dir', type=str, required=False, default=None,
                    help='Path to the root data directory that contains the images.')
parser.add_argument('--latent_dir', type=str, required=False, default=None,
                    help='Path to the root latent directory that contains the latents.')
parser.add_argument('--diff_ckpt', type=str, required=False, default=None,
                    help='Path to the AEKL checkpoint that contains the autoencoder and latent diffusion model.')
parser.add_argument('--num_channel', type=int, required=False, default=16,
                    help='Number of channels in the latent space.')
parser.add_argument('--gpu', type=str, default='0',
                    help='Specify the GPU ids to use, e.g., "0,1,2,3"')
parser.add_argument('--checkpoints_path', type=str, required=False, default='./checkpoints',
                    help='Path to save model checkpoints.')
parser.add_argument('--DEBUG', action='store_true', help='If set, run in debug mode with fewer data.')

parser.add_argument('--input_modality', nargs='+', required=True, help='List of modalities (e.g., t1c t1n t2w t2f)')
parser.add_argument('--target_modality', nargs='+', required=True, help='List of modalities (e.g., t1c t1n t2w t2f)')

parser.add_argument('--use_standard_norm', action='store_true', help='If set, use standard normalization for input and output images.')

parser.add_argument('--inference', action='store_true')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
os.environ["WANDB_MODE"] = "disabled"

import time
import torchvision
import matplotlib.cm as cm
from collections import defaultdict
import torch
import time
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import Tensor
from generative.networks.nets import DiffusionModelUNet
from tqdm import tqdm
import numpy as np


# -------------- Norm ----------
use_standard_norm = args.use_standard_norm
if use_standard_norm:
    args.checkpoints_path += "_stdnorm"

def standard_normalize(x, mean=None, std=None, eps=1e-6):
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



# -------------- Generation function ----------


def euler_step(model: DiffusionModelUNet, x_t: Tensor, t_start: Tensor, t_end: Tensor):
    # delta_t shape (B, 1, 1, 1)
    delta_t = (t_end - t_start).view(-1, 1, 1, 1)

    # model si aspetta t come tensor (B,)
    v_hat = model(x_t, t_start)

    x_t_noise = x_t[:, len(args.input_modality):, :, :]  # [B, 1, H, W]
    x_t_cond  = x_t[:, :len(args.input_modality), :, :]  # [B, 1, H, W], che è T1

    x_next_noise = x_t_noise + delta_t * v_hat

    x_next = torch.cat([x_t_cond, x_next_noise], dim=1)  # [B, 2, H, W]
    return x_next


@torch.no_grad()
def generate(model: nn.Module, condition: Tensor, gen_steps: int = 20):
    model.eval()

    device = condition.device
    batch_size = condition.shape[0]

    time_steps = torch.linspace(
        0.0, 1.0, gen_steps + 1, device=device, dtype=torch.float32)

    x = condition


    for i in range(gen_steps):
        t_start = time_steps[i].expand(batch_size)
        t_end   = time_steps[i + 1].expand(batch_size)
        x = euler_step(model, x_t=x, t_start=t_start, t_end=t_end)
        
        # print("cond:", condition.shape, x.shape)


    return x[:, len(args.input_modality):, :, :]  # return only the generated part



# ------------------ Inference
def flow_matching_inference(model, x0, n_steps=20, device="cuda"):
    """
    Integrates from x0 at t=0 to x1 at t=1 using the flow-matching model.
    """
    model.eval()
    B = x0.shape[0]
    x_t = x0.clone()
    t_points = torch.linspace(0, 1, n_steps+1, device=device)

    with torch.no_grad():
        for i in range(n_steps):
            t0 = t_points[i].repeat(B).to(device)  # [B]
            t_img = t0.view(B, 1, 1, 1)
            v_t = model(x_t, t0)   # predicted velocity [B, 1, H, W]
            dt = t_points[i+1] - t_points[i]
            x_t = x_t + v_t * dt   # Euler integration step

    return x_t  # approximation of x1


import torch.nn.functional as F


def downsample(img, scale_factor):
    return F.interpolate(img, scale_factor=scale_factor, mode="bilinear", align_corners=False)

def multi_scale_loss(src, tgt):
    """
    pred_flows: list of flows [F_low, F_mid, F_high]
    src: source image (B, C, H, W)
    tgt: target image (B, C, H, W)
    """
    loss = 0.0
    scales = [0.25, 0.5, 1.0]
    weights = [0.5, 0.7, 1.0]  # weight more at higher resolution
    
    for i, flow in enumerate(scales):
        # Downsample images to current scale
        src_s = downsample(src, scales[i])
        tgt_s = downsample(tgt, scales[i])

        # Warp source image using flow
        # warped_src = warp(src_s, flow)

        # Reconstruction loss (L1 + SSIM)
        # l1_loss = torch.mean(torch.abs(src_s - tgt_s))
        l1_loss = nn.MSELoss()(src_s, tgt_s)
        # ssim_loss = 1 - ssim(warped_src, tgt_s)  # define ssim() separately

        loss += weights[i] * (l1_loss ) # + 0.5 * ssim_loss)

    return loss


# ------------------ Loss 
def nontriviality_loss(flow: torch.Tensor) -> torch.Tensor:
    var = torch.var(flow, dim=(1, 2, 3), unbiased=False)
    # mean_mag = flow.abs().mean(dim=(1, 2, 3)) + 1e-8
    return -(var).mean()



# ------------------ Training function ----------
import torch
print("Using", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


def train_flow(model: DiffusionModelUNet, device: str, train_loader: DataLoader, 
               val_loader: DataLoader, project: str, exp_name: str, notes: str, n_epochs: int = 10, lr: float = 1e-3, 
               generation_steps: int = 100, val_gen_steps=1):
    
    CHECKPOINTS_PATH = args.checkpoints_path + "/checkpoints"
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
    IMAGE_OUTPUT_PATH = args.checkpoints_path + "/image_output"
    os.makedirs(IMAGE_OUTPUT_PATH, exist_ok=True)

   

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=args.lr_min)

    best_val_loss = float("inf")
    best_model_path = None
    start_time = time.time()
    best_val_psnr = 0.0
    
    for e in range(1, n_epochs + 1):
        start_e_time = time.time()
        # Training
        model.train()
        train_losses = []

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {e}')

        total_loss = 0
        for step, batch in progress_bar:
            if args.DEBUG and step > 10:
                break

            if step > 500:
                break
            
            DCE1 = batch["DCE1"].to(device)  # [B, 1, H, W]
            DCE2 = batch["DCE2"].to(device)
            DCE3 = batch["DCE3"].to(device)

            if use_standard_norm:
                DCE1, mean_1, std_1 = standard_normalize(DCE1)
                DCE2, mean_2, std_2 = standard_normalize(DCE2)
                DCE3, mean_3, std_3 = standard_normalize(DCE3)

            
            x_0 = []
            for mod in args.input_modality:
                if   mod == "DCE1":  x_0.append(DCE1)
                elif mod == "DCE2":  x_0.append(DCE2)
                elif mod == "DCE3":  x_0.append(DCE3)
            x_0 = torch.cat(x_0, dim=1)  # [B

            x_1 = []
            for mod in args.target_modality:
                if   mod == "DCE1":   x_1.append(DCE1)
                elif mod == "DCE2":  x_1.append(DCE2)
                elif mod == "DCE3":  x_1.append(DCE3)
            x_1 = torch.cat(x_1, dim=1)  # [B

            B = x_0.shape[0]
            t = torch.rand(B, device=device)  # B
            t_img = t.view(B, 1, 1, 1)  # [B, 1, 1, 1] for broadcasting

            if x_0.shape[1] > 1:
                # average across channels -> [B,1,H,W]
                x_0_c = x_0.mean(dim=1, keepdim=True)
                x_0_c = x_0_c.repeat(1, x_1.shape[1], 1, 1)  # [B,C,H,W]
            else:
                x_0_c = x_0
                x_0_c = x_0_c.repeat(1, x_1.shape[1], 1, 1)  # [B,C,H,W]


            x_t  = (1 - t_img) * x_0_c + t_img * x_1         # [B, 1, H, W]
            dx_t = x_1 - x_0_c                              # [B, 1, H, W]
            dx_t = dx_t.detach()
            x_t = x_t.detach()

            x_t_input = torch.cat([x_0, x_t], dim=1)  # [B, 2, H, W] or [B, 3, H, W]

            optimizer.zero_grad()
            pred = model(x_t_input, t)  # [B, 1, H, W]

            # loss = criterion(pred, dx_t)         # LOss
            loss = multi_scale_loss(pred, dx_t) # + 0.01 * nontriviality_loss(pred)

            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            

            total_loss += loss.item()
            avg_loss   = total_loss / (step + 1)
            current_lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix(loss=avg_loss, lr=current_lr)


        lr_scheduler.step()


        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim

        val_losses, val_psnr, val_ssim = [], {}, {}
        for c, mod_name in enumerate(args.target_modality):
            val_psnr[mod_name] = []
            val_ssim[mod_name] = []
            
        # Geometric warps: small affine/TPS warps, elastic deformations → force FM to align structures, not just pixels.

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():

            val_num = 50
            progress_bar = tqdm(val_loader, desc="Test")

            for step, batch in enumerate(progress_bar):
                if args.DEBUG and step > 10 and not args.inference:
                    break

                if step > val_num  and not args.inference:
                    break
            
                if args.inference and step > 500:
                    break
                         
                DCE1 = batch["DCE1"].to(device)  # [B,1,H,W]
                DCE2 = batch["DCE2"].to(device)
                DCE3 = batch["DCE3"].to(device)

                # --- normalization ---
                mod_stats = {}  # store mean/std for each modality
                if use_standard_norm:
                    DCE1, mean_1, std_1 = standard_normalize(DCE1)
                    mod_stats["DCE1"] = (mean_1, std_1)

                    DCE2, mean_2, std_2 = standard_normalize(DCE2)
                    mod_stats["DCE2"] = (mean_2, std_2)

                    DCE3, mean_3, std_3 = standard_normalize(DCE3)
                    mod_stats["DCE3"] = (mean_3, std_3)

                # --- input modalities ---
                x_0 = []
                for mod in args.input_modality:
                    if   mod == "DCE1":  x_0.append(DCE1)
                    elif mod == "DCE2":  x_0.append(DCE2)
                    elif mod == "DCE3":  x_0.append(DCE3)
                x_0 = torch.cat(x_0, dim=1)  # [B,n,H,W]

                # --- target modalities ---
                x_1 = []
                target_names = []
                for mod in args.target_modality:
                    if   mod == "DCE1":  x_1.append(DCE1)
                    elif mod == "DCE2":  x_1.append(DCE2)
                    elif mod == "DCE3":  x_1.append(DCE3)
                    target_names.append(mod)
                x_1 = torch.cat(x_1, dim=1)  # [B,m,H,W]

                # --- interpolation ---
                B = x_0.shape[0]
                t = torch.rand(B, device=device)       # [B]
                t_img = t.view(B, 1, 1, 1)             # [B,1,1,1]

                if x_0.shape[1] > 1:
                    x_0_c = x_0.mean(dim=1, keepdim=True)
                    x_0_c = x_0_c.repeat(1, x_1.shape[1], 1, 1)
                else:
                    x_0_c = x_0
                    x_0_c = x_0_c.repeat(1, x_1.shape[1], 1, 1)

                x_t = (1 - t_img) * x_0_c + t_img * x_1  # [B,m,H,W]
                x_0_c = x_0_c.detach()
                x_t   = x_t.detach()

                # --- model input ---
                x_t_input = torch.cat([x_0, x_0_c], dim=1)  # [B,n+m,H,W]
                # x_t_input = torch.cat([x_0, x_t], dim=1)  # [B,n+m,H,W]
                x_hat = generate(model=model, condition=x_t_input, gen_steps=val_gen_steps)

                # print("x_hat:", x_hat.shape, "x_1:", x_1.shape)

                # --- denormalize per modality ---
                if use_standard_norm:
                    x_hat_list, x1_list = [], []
                    for c, mod_name in enumerate(target_names):
                        mean, std = mod_stats[mod_name]
                        
                        x_1_img = x_1[:, c:c+1]
                        x_hat_img = x_hat[:, c:c+1]

                        # if use_standard_norm:
                        #     x_hat_img = standard_normalize(x_hat_img)[0]
                        #     x_hat_img = x_hat_img if torch.abs(x_1_img - x_hat_img).mean() < torch.abs(x_1_img - x_hat[:, c:c+1]).mean() else x_hat[:, c:c+1]


                        x_hat_list.append(denormalize(x_hat_img, mean, std))
                        x1_list.append(   denormalize(x_1_img, mean, std))

                    x_hat = torch.cat(x_hat_list, dim=1)
                    x_1   = torch.cat(x1_list, dim=1)

                # --- evaluation ---
                x_hat_np = x_hat.cpu().numpy()
                x1_np    = x_1.cpu().numpy()

                for c, mod_name in enumerate(target_names):
                    for i in range(B):
                        
                        out_img = x_hat_np[i, c]
                        gt_img  = x1_np[i, c]

                        out_img = np.clip(out_img, gt_img.min(), gt_img.max())

                        
                        psnr_val = psnr(gt_img, out_img, data_range=1.0)
                        ssim_val = ssim(gt_img, out_img, data_range=1.0)
                        val_psnr[mod_name].append(psnr_val)
                        val_ssim[mod_name].append(ssim_val)

            e_time = time.time() - start_e_time

            # --- averages ---
            for mod_name in target_names:
                avg_val_psnr = sum(val_psnr[mod_name]) / len(val_psnr[mod_name])
                avg_val_ssim = sum(val_ssim[mod_name]) / len(val_ssim[mod_name])
                print(f"Time {e}: {e_time:.2f}, Modality {mod_name}, "
                    f"PSNR: {avg_val_psnr:.2f}, SSIM: {avg_val_ssim:.4f}")




        
        # ---------- Checkpoint for visualization ----------
        batch = next(iter(val_loader))  # just one batch

        # --- utility: colormap for residuals ---
        def apply_colormap(tensor, cmap="magma"):
            """
            Convert a single-channel tensor [1,H,W] to [3,H,W] RGB using a matplotlib colormap.
            """
            tensor = tensor.detach().cpu().squeeze().numpy()   # [H,W]
            # tensor = tensor 
            # tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
            colored = cm.get_cmap(cmap)(tensor)[..., :3]       # [H,W,3]
            colored = torch.from_numpy(colored).permute(2, 0, 1).float()  # [3,H,W]
            return colored


        # --- training/eval loop (snippet) ---
        DCE1_ori = batch["DCE1"].to(device)  # [B,1,H,W]
        DCE2_ori = batch["DCE2"].to(device)
        DCE3_ori = batch["DCE3"].to(device)

        # --- normalization ---
        mod_stats = {}
        if use_standard_norm:
            DCE1, mean_1, std_1 = standard_normalize(DCE1_ori)
            mod_stats["DCE1"] = (mean_1, std_1)

            DCE2, mean_2, std_2 = standard_normalize(DCE2_ori)
            mod_stats["DCE2"] = (mean_2, std_2)

            DCE3, mean_3, std_3 = standard_normalize(DCE3_ori)
            mod_stats["DCE3"] = (mean_3, std_3)

        # --- build input modalities ---
        x_0 = []
        for mod in args.input_modality:
            if   mod == "DCE1":  x_0.append(DCE1)
            elif mod == "DCE2":  x_0.append(DCE2)
            elif mod == "DCE3":  x_0.append(DCE3)
        x_0 = torch.cat(x_0, dim=1)  # [B,n,H,W]

        # --- build target modalities ---
        x_1 = []
        target_names = []
        for mod in args.target_modality:
            if   mod == "DCE1": x_1.append(DCE1)
            elif mod == "DCE2": x_1.append(DCE2)
            elif mod == "DCE3": x_1.append(DCE3)
            target_names.append(mod)
        x_1 = torch.cat(x_1, dim=1)  # [B,m,H,W]

        # --- interpolation ---
        B = x_0.shape[0]
        t = torch.rand(B, device=device)       # [B]
        t_img = t.view(B, 1, 1, 1)             # [B,1,1,1]

        if x_0.shape[1] > 1:
            x_0_c = x_0.mean(dim=1, keepdim=True)
            x_0_c = x_0_c.repeat(1, x_1.shape[1], 1, 1)
        else:
            x_0_c = x_0
            x_0_c = x_0_c.repeat(1, x_1.shape[1], 1, 1)

        x_0_c = x_0_c.detach()

        # --- model input ---
        x_t_input = torch.cat([x_0, x_0_c], dim=1)  # [B,n+m,H,W]
        CTC_gen   = generate(model=model, condition=x_t_input, gen_steps=generation_steps if not args.DEBUG else val_gen_steps)

        # --- denormalize per modality ---
        if use_standard_norm:
            gt_list, gen_list = [], []
            for c, mod_name in enumerate(target_names):
                mean, std = mod_stats[mod_name]

                x_1_aa = CTC_gen[:, c:c+1]
                x_1_aa = standard_normalize(x_1_aa)[0]
                
                gt_list.append(denormalize(x_1[:, c:c+1], mean, std))
                gen_list.append(denormalize(x_1_aa, mean, std))
                
            x_1 = torch.cat(gt_list, dim=1)       # [B,m,H,W]
            CTC_gen = torch.cat(gen_list, dim=1)  # [B,m,H,W]

        # ==================================================================
        # --- SAVE IMAGE GRID (first sample in batch) ---
        # ==================================================================
        i = 0
        imgs_to_show = []

        # input modalities (DCE1, DCE2, DCE3)
        for mod in args.input_modality:
            if mod == "DCE1":
                imgs_to_show.append(DCE1_ori[i].repeat(3,1,1).cpu())
            elif mod == "DCE2":
                imgs_to_show.append(DCE2_ori[i].repeat(3,1,1).cpu())
            elif mod == "DCE3":
                imgs_to_show.append(DCE3_ori[i].repeat(3,1,1).cpu())

        # target modalities: GT, Gen, Residual
        for c, mod_name in enumerate(target_names):
            gt_img  = x_1[i, c:c+1]       # [1,H,W]
            gen_img = CTC_gen[i, c:c+1]   # [1,H,W]

            # grayscale → RGB
            imgs_to_show.append(gt_img.repeat(3,1,1).cpu())
            imgs_to_show.append(gen_img.repeat(3,1,1).cpu())

            # residual map (colormap)
            residual = (gen_img - gt_img).abs()
            residual_colored = apply_colormap(residual)
            imgs_to_show.append(residual_colored)

        # make grid (nrow = number of images in row)
        nrow = len(imgs_to_show)  # all in one row
        grid = torchvision.utils.make_grid(imgs_to_show, nrow=nrow, pad_value=1)

        # save
        out_path = f"{IMAGE_OUTPUT_PATH}/{exp_name}_epoch{e+1}.png"
        torchvision.utils.save_image(grid, out_path)
        print(f"✅ Saved visualization → {out_path}")


        if e % 2 == 0 or e == n_epochs - 1 or avg_val_psnr > best_val_psnr or args.DEBUG:
            
                 
            if best_val_psnr < avg_val_psnr:
                if args.DEBUG:
                    path = f'{CHECKPOINTS_PATH}/checkpoint_{exp_name}_debug_best.pth'
                else:
                    path = f'{CHECKPOINTS_PATH}/checkpoint_{exp_name}_{e+1}_best.pth'
            
                
                torch.save({
                    'epoch': e + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
                if best_model_path is not None and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                best_model_path = path
                best_val_psnr = avg_val_psnr
                
            
            print("Saved checkpoint to", path)
            

        end_time = time.time()
        elapsed_time = end_time - start_time
       
        print(
            f"Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")
    print("Training complete.")
    return best_model_path


def main():
    OUTPUT_DIR       = args.checkpoints_path + "/outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    
    spatial_size   = [256, 256]  # 512 or 256
    target_dataset = ["Adrenal"]
    
    from src.dce_2D_dataloader import create_triplet_dataloader
    train_loader, train_ds     = create_triplet_dataloader(csv_path=args.dataset_csv, resolution=256, split="train", 
                                                          batch_size=args.batch_size, spatial_size=spatial_size)
    
    spatial_size = (256, 256)  
    val_loader,  test_ds      = create_triplet_dataloader(csv_path=args.dataset_csv, resolution=256, split="test",  
                                                          batch_size=4, spatial_size=spatial_size)



    model = DiffusionModelUNet(
        spatial_dims=2,  # 2D
        in_channels=len(args.input_modality) + len(args.target_modality),   # x
        out_channels=len(args.target_modality),                         # predice delta_x_t
    )

    if args.diff_ckpt:
        ckpt = torch.load(args.diff_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"✅ Loaded checkpoint from {args.diff_ckpt}")


    # ---------- Model training ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = f"unetflow-CTCTC-s{args.n_epochs}e"
    prediction_dir = f'{OUTPUT_DIR}/{exp_name}'
    project_name = 'Medical-I2I-Benchmark'

    best_modelpath = train_flow(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        project=project_name,
        exp_name=exp_name,
        notes="Small UNet flow model for directional diffusion from CT to CTC.",
        n_epochs=args.n_epochs,
        lr=args.lr,
        generation_steps=300)

    # Load the best checkpoint
    model.load_state_dict(torch.load(best_modelpath, map_location=device)['model_state_dict'])
    model.eval()



if __name__ == "__main__":
    main()

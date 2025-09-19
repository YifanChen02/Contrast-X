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
parser.add_argument('--aekl_ckpt', type=str, required=False, default=None,
                    help='Path to the AEKL checkpoint that contains the autoencoder and latent diffusion model.')
parser.add_argument('--num_channel', type=int, required=False, default=16,
                    help='Number of channels in the latent space.')
parser.add_argument('--gpu', type=str, default='0',
                    help='Specify the GPU ids to use, e.g., "0,1,2,3"')
parser.add_argument('--checkpoints_path', type=str, required=False, default='./checkpoints',
                    help='Path to save model checkpoints.')
parser.add_argument('--DEBUG', action='store_true', help='If set, run in debug mode with fewer data.')


parser.add_argument('--use_standard_norm', action='store_true', help='If set, use standard normalization for input and output images.')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
os.environ["WANDB_MODE"] = "disabled"

import torch
import time
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import Tensor
from generative.networks.nets import DiffusionModelUNet
from tqdm import tqdm



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

    x_t_noise = x_t[:, 0:1, :, :]  # [B, 1, H, W]
    x_t_cond = x_t[:, 1:2, :, :]  # [B, 1, H, W], che è T1

    x_next_noise = x_t_noise + delta_t * v_hat

    x_next = torch.cat([x_next_noise, x_t_cond], dim=1)  # [B, 2, H, W]
    return x_next




@torch.no_grad()
def generate(model: nn.Module, condition: Tensor, gen_steps: int = 20):
    model.eval()

    device = condition.device
    batch_size = condition.shape[0]

    time_steps = torch.linspace(0.0, 1.0, gen_steps + 1, device=device, dtype=torch.float32)

    x = torch.cat([torch.randn_like(condition, device=device), condition], dim=1)  # [B, 2, H, W]
    for i in range(gen_steps):
        t_start = time_steps[i].expand(batch_size)
        t_end = time_steps[i + 1].expand(batch_size)
        x = euler_step(model, x_t=x, t_start=t_start, t_end=t_end)

    return x[:, 0:1, :, :]  # [B, 1, H, W]





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
        l1_loss = torch.mean(torch.abs(src_s - tgt_s))
        # ssim_loss = 1 - ssim(warped_src, tgt_s)  # define ssim() separately

        loss += weights[i] * (l1_loss ) # + 0.5 * ssim_loss)

    return loss


# ------------------ Loss 
def nontriviality_loss(flow: torch.Tensor) -> torch.Tensor:
    var = torch.var(flow, dim=(1, 2, 3), unbiased=False)
    # mean_mag = flow.abs().mean(dim=(1, 2, 3)) + 1e-8
    return -(var).mean()



# ------------------ Training function ----------


def train_flow(model: DiffusionModelUNet, device: str, train_loader: DataLoader, 
               val_loader: DataLoader, project: str, exp_name: str, notes: str, n_epochs: int = 10, lr: float = 1e-3, generation_steps: int = 100):
    
    CHECKPOINTS_PATH = args.checkpoints_path + "/checkpoints"
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
    IMAGE_OUTPUT_PATH = args.checkpoints_path + "/image_output"
    os.makedirs(IMAGE_OUTPUT_PATH, exist_ok=True)

    print("Using", torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else "CPU")

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
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

            x_1 = batch["CTC"].to(device)  # [B, 1, H, W]
            x_0 = batch["CT"].to(device)

            x_0_noise = torch.randn_like(x_0).to(device)  # [B, 1, H, W]

            if use_standard_norm:
                x_0, mean_0, std_0 = standard_normalize(x_0)
                x_1, mean_1, std_1 = standard_normalize(x_1)


            B = x_0.shape[0]
            t = torch.rand(B, device=device)  # B

            t_img = t.view(B, 1, 1, 1)  # [B, 1, 1, 1] for broadcasting

            x_t = (1 - t_img) * x_0_noise + t_img * x_1  # [B, 1, H, W]
            x_t = torch.cat([x_t, x_0], dim=1)  # [B, 2, H, W]

            # x_t  = (1 - t_img) * x_0 + t_img * x_1         # [B, 1, H, W]
            dx_t = x_1 - x_0_noise                              # [B, 1, H, W]

            optimizer.zero_grad()
            pred = model(x_t, t)  # [B, 1, H, W]
            # loss = criterion(pred, dx_t)

            loss = multi_scale_loss(pred, dx_t) + 0.01 * nontriviality_loss(pred)
            
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)
            progress_bar.set_postfix(loss=avg_loss)




        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim

        val_losses, val_psnr, val_ssim = [], [], []


        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():

            val_num = 50
            progress_bar = tqdm(train_loader, desc="Test")

            for step, batch in enumerate(progress_bar):
                if args.DEBUG and step > 10:
                    break

                if step > val_num:
                    break

                x_1 = batch["CTC"].to(device)
                x_0 = batch["CT"].to(device)
                if use_standard_norm:
                    x_0, mean_0, std_0 = standard_normalize(x_0)
                    x_1, mean_1, std_1 = standard_normalize(x_1)

                B = x_0.shape[0]
                t = torch.rand(B, device=device)
                t_img = t.view(B, 1, 1, 1)
                x_hat = generate(model=model, condition=x_0, gen_steps=20)
                if use_standard_norm:
                    x_hat = denormalize(x_hat, mean_1, std_1)
                    x_1   = denormalize(x_1,   mean_1, std_1)

                x_hat_np = x_hat.cpu().numpy()
                x1_np    = x_1.cpu().numpy()
                for i in range(B):
                    out_img = x_hat_np[i, 0]
                    gt_img  = x1_np[i, 0]
                    val_psnr.append(psnr(gt_img, out_img, data_range=gt_img.max() - gt_img.min()))
                    val_ssim.append(ssim(gt_img, out_img, data_range=gt_img.max() - gt_img.min()))




        e_time = time.time() - start_e_time

        avg_val_psnr   = sum(val_psnr) / len(val_psnr)
        avg_val_ssim   = sum(val_ssim) / len(val_ssim)
        
        print(f"Time {e}: {e_time:.2f}, Val PSNR: {avg_val_psnr:.2f}, SSIM: {avg_val_ssim:.4f}")


        lr_scheduler.step()
        # Checkpoint

        sample_batch = next(iter(val_loader))  # just one batch
        CT_gt   = sample_batch["CT"][0].unsqueeze(0).to(device)
        CTC_gt  = sample_batch["CTC"][0].to(device)  # [1, 1, H, W]
        if use_standard_norm:
            CT_gt, mean_0, std_0 = standard_normalize(CT_gt)
            CTC_gt, mean_1, std_1 = standard_normalize(CTC_gt)
        CTC_gen = generate(model=model, condition=CT_gt,
                            gen_steps=generation_steps)

        if use_standard_norm:
            CT_gt = denormalize(CT_gt, mean_0, std_0)
            CTC_gt = denormalize(CTC_gt, mean_1, std_1)
            CTC_gen = denormalize(CTC_gen, mean_1, std_1)

        
        # residuals (keep signed values for colormap)
        # Residuals (absolute difference)
        residual_gt  = (CTC_gt - CT_gt.squeeze(0)).abs()
        residual_gen = (CTC_gen.squeeze(0) - CT_gt.squeeze(0)).abs()

        import matplotlib.cm as cm
        def apply_colormap(tensor, cmap="magma"):
            """
            Convert a single-channel tensor [H, W] to a 3-channel RGB colormap image.
            """
            tensor = tensor.detach().cpu().squeeze().numpy()  # ✅ make sure it's [H,W]
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)  # normalize 0-1
            colored = cm.get_cmap(cmap)(tensor)[..., :3]  # [H,W,3], drop alpha
            colored = torch.from_numpy(colored).permute(2, 0, 1).float()  # [3,H,W]
            return colored


        residual_gt_colored = apply_colormap(residual_gt, cmap="magma")
        residual_gen_colored = apply_colormap(residual_gen, cmap="magma")

        imgs = [CT_gt.squeeze(0).cpu(), 
                CTC_gt.cpu(),  
                residual_gt_colored.cpu(),      
                CTC_gen.squeeze(0).cpu(),
                residual_gen_colored.cpu()          # colored
            ] # [3, 1, H, W]

        imgs = torch.stack([img.repeat(3,1,1) if img.ndim==3 and img.shape[0]==1 else img for img in imgs], dim=0)  # [N, 3, H, W]

        grid = torchvision.utils.make_grid(imgs, nrow=5)
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

    
 
    spatial_size   = [128, 128]  # 512 or 256
    target_dataset = ["Adrenal"]
    
    from src.ct_2D_dataloader import create_paired_dataloader

    train_loader, train_ds     = create_paired_dataloader(csv_path=args.dataset_csv, resolution=256, split="train", 
                                                          batch_size=args.batch_size, spatial_size=spatial_size)
    
    spatial_size = (256, 256)  
    val_loader,  test_ds      = create_paired_dataloader(csv_path=args.dataset_csv, resolution=256, split="test",  
                                                          batch_size=4, spatial_size=spatial_size)

    model = DiffusionModelUNet(
        spatial_dims=2,  # 2D
        in_channels=2,  # x
        out_channels=1,  # predice delta_x_t
    )

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


    # ---------- Model evaluation and prediction generation ----------
    from t1t2converter.utils import CHECKPOINTS_PATH, DATAPATH, OUTPUT_DIR, compute_ssim_from_dataset, generate_and_save_predictions, normalize_image


if __name__ == "__main__":
    main()



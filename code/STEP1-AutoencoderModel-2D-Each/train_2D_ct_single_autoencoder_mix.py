import os, gc, sys
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils import args
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


import torch
import warnings
import numpy as np
import torch
from tqdm import tqdm
from monai.utils import set_determinism

from torch.nn import L1Loss
# from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from monai.losses import PerceptualLoss, PatchAdversarialLoss
import torch.nn.functional as F


from utils import args, import_from_dotted_path, utils_metric
from src import utils_usage
from monai.losses import PerceptualLoss

from src import init_autoencoder, KLDivergenceLoss, init_patch_discriminator
from utils.utils_image import save_image, pad_to_shape

from src.dce_2D_dataloader import create_triplet_dataloader
from accelerate import Accelerator

from src.ct_2D_dataloader import create_paired_dataloader
torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore")

set_determinism(0)
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_epoch = 1
import torch.nn as nn

accelerator = Accelerator()
DEVICE = accelerator.device


# Ratio to drop
valid_ratio = 0  # All mask
train_ratio = 0  # 0.5  # Half mask

use_standard_norm = args.use_standard_norm  # use_standard_norm
use_broken        = args.use_broken  # True

print(" === use_standard_norm:", use_standard_norm)

ACTIVATION_CLASSES = (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.PReLU, nn.RReLU)

from skimage.metrics import peak_signal_noise_ratio as psnr_metric, structural_similarity as ssim_metric
import numpy as np

import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def print_model_shapes(model, input_size):

    x = torch.randn(input_size, dtype=torch.float16).to(DEVICE)
    hooks = []

    def hook_fn(module, input, output):
        if isinstance(output, (list, tuple)):
            out_shape = [o.shape for o in output]
        else:
            out_shape = output.shape
        print(f"{module.__class__.__name__:<30} | Input: {input[0].shape} -> Output: {out_shape}")

    for name, layer in model.named_modules():
        if layer != model:
            hooks.append(layer.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        with accelerator.autocast():
            model(x)

    for h in hooks:
        h.remove()


def to_numpy_image(tensor):
    # Assumes input: [1, H, W] or [H, W]
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().squeeze()
        array = tensor.numpy()
    else:
        array = tensor
    return array


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

            
def batch_psnr_ssim(imgs1: np.ndarray, imgs2: np.ndarray, data_range=1.0):
    """
    Compute PSNR and SSIM for batches of images in (B, C, H, W) format.

    Parameters:
        imgs1, imgs2: np.ndarray of shape (B, C, H, W)
        data_range: max pixel value range (1.0 if normalized, 255 for uint8)

    Returns:
        psnr_vals: list of PSNR values (per image)
        ssim_vals: list of SSIM values (per image)
    """
    B = imgs1.shape[0]
    psnr_vals, ssim_vals = [], []

    for b in range(B):
        # Move channel axis to last: (H, W, C)
        img1 = np.moveaxis(imgs1[b], 0, -1)
        img2 = np.moveaxis(imgs2[b], 0, -1)

        psnr_vals.append(psnr_metric(img1, img2, data_range=data_range))
        ssim_vals.append(ssim_metric(img1, img2, data_range=data_range, channel_axis=-1))

    return psnr_vals, ssim_vals


def validate_model(model, dataloader, device, image_save_root=None, max_batches=6,
                       step_name="", image_key="source", image_key_str=None):
    model.eval()
    if image_save_root is not None:
        os.makedirs(image_save_root, exist_ok=True)

    # metrics for "reconstruction" (with dropout)
    avg_psnr_dce1, avg_ssim_dce1 = [], []
    avg_psnr_dce2, avg_ssim_dce2 = [], []
    avg_psnr_dce3, avg_ssim_dce3 = [], []

    avg_psnr_dce1_mask, avg_ssim_dce1_mask = [], []
    avg_psnr_dce2_mask, avg_ssim_dce2_mask = [], []
    avg_psnr_dce3_mask, avg_ssim_dce3_mask = [], []

    # metrics for "clean reconstruction" (full input)
    avg_psnr_clean_dce1, avg_ssim_clean_dce1 = [], []
    avg_psnr_clean_dce2, avg_ssim_clean_dce2 = [], []
    avg_psnr_clean_dce3, avg_ssim_clean_dce3 = [], []



    with torch.no_grad():
        for idx, batch in enumerate(dataloader):


            if idx >= max_batches:
                break

            dce1 = batch['CT'].to(device)
            dce2 = batch['CTC'].to(device)
            images = torch.cat([dce1, dce2], dim=1)
            
            if use_standard_norm:
                dce1, dce1_mean, dce1_std    = standard_normalize(dce1)
                dce2, dce2_mean, dce2_std    = standard_normalize(dce2)


            # --- broken input (simulate missing DCE2/DCE3) ---
            inputs, mask = [dce1, torch.zeros_like(dce1).to(device)], [True, False]  # DCE1 always present
   

            # print("mask = ", mask)

            # --- clean input (all modalities present) ---
            clean_inputs = [dce1, dce2]
            clean_mask   = [True, True]

            with accelerator.autocast():
                clean_recon = model(clean_inputs, modalities_mask=clean_mask,
                                    use_multiple_encoders=True).sample
                out = model(inputs, modalities_mask=mask, use_multiple_encoders=True)
                reconstruction = out.sample

            # --- Move to CPU for metrics ---
            
            if use_standard_norm:
                recon_dce1 = denormalize(reconstruction[:, 0:1], dce1_mean, dce1_std)
                recon_dce2 = denormalize(reconstruction[:, 1:2], dce2_mean, dce2_std)
                reconstruction = torch.cat([recon_dce1, recon_dce2], dim=1)

                # clean recon
                clean_dce1 = denormalize(clean_recon[:, 0:1], dce1_mean, dce1_std)
                clean_dce2 = denormalize(clean_recon[:, 1:2], dce2_mean, dce2_std)
                clean_recon = torch.cat([clean_dce1, clean_dce2], dim=1)
            else:
                reconstruction = torch.clamp(reconstruction, 0, 1)
                clean_recon    = torch.clamp(clean_recon, 0, 1)


            image_np       = images.cpu().numpy()
            recon_np       = reconstruction.cpu().numpy()
            clean_recon_np = clean_recon.cpu().numpy()

            

            # ---- save images if needed ----
            if image_save_root is not None:
                for b in range(image_np.shape[0]):
                    orig  = image_np[b]
                    recon = recon_np[b]
                    clean = clean_recon_np[b]

                    combined_slices = []
                    for i in range(image_np.shape[1]):
                        combined_slices.append(
                            np.concatenate([orig[i], recon[i], clean[i]], axis=0)
                        )
                    combined = np.concatenate(combined_slices, axis=1)

                    save_path = os.path.join(image_save_root, f"{step_name}_img_{idx}_{b}.jpg")
                    save_image(save_path, combined)

            # ---- metrics for broken input (split mask/unmask) ----
            for ch, (avg_psnr_unmask, avg_ssim_unmask, avg_psnr_mask, avg_ssim_mask) in enumerate([
                (avg_psnr_dce1, avg_ssim_dce1, avg_psnr_dce1_mask, avg_ssim_dce1_mask),
                (avg_psnr_dce2, avg_ssim_dce2, avg_psnr_dce2_mask, avg_ssim_dce2_mask)
            ]):
                psnr, ssim = batch_psnr_ssim(image_np[:, ch:ch+1],
                                            recon_np[:, ch:ch+1],
                                            data_range=1.0)
                
                
                avg_psnr_unmask.append(psnr)
                avg_ssim_unmask.append(ssim)
                

            # ---- metrics for clean input ----
            for ch, (avg_psnr, avg_ssim) in enumerate([
                (avg_psnr_clean_dce1, avg_ssim_clean_dce1),
                (avg_psnr_clean_dce2, avg_ssim_clean_dce2),
            ]):
                psnr, ssim = batch_psnr_ssim(image_np[:, ch:ch+1],
                                            clean_recon_np[:, ch:ch+1],
                                            data_range=1.0)
                avg_psnr.append(psnr)
                avg_ssim.append(ssim)

    # ---- final logs ----
    print("\n" + "="*60)
    print(f"{step_name} - Metrics Summary")
    print("="*60)
    print(f"{'Modality':<12} {'Condition':<10} {'PSNR':>8} {'SSIM':>10}")
    print("-"*60)

    # DCE1
    print(f"{'CT':<12} {'FULL':<10} {np.mean(avg_psnr_dce1):>8.2f} {np.mean(avg_ssim_dce1):>10.4f}")
    print(f"{'CTC':<12} {'MASK':<10} {np.mean(avg_psnr_dce2):>8.2f} {np.mean(avg_ssim_dce2):>10.4f}")

    # CLEAN (always available, so no mask/unmask split)
    print(f"{'CLEAN CT':<12} {'FULL':<10} {np.mean(avg_psnr_clean_dce1):>8.2f} {np.mean(avg_ssim_clean_dce1):>10.4f}")
    print(f"{'CLEAN CTC':<12} {'FULL':<10} {np.mean(avg_psnr_clean_dce2):>8.2f} {np.mean(avg_ssim_clean_dce2):>10.4f}")
    print("="*60 + "\n")





message = ""

missing_modality = args.missing_modality

# from MM_AE import AutoencoderKL_multi_encoder
from SM_AE import AutoencoderKL_single_encoder

def define_2DAE(in_channels=3,  out_channels=2, latent_channels = 16):
    from huggingface_hub import snapshot_download
    from diffusers import AutoencoderKL

    
    from diffusers.models import AutoencoderKL
    # block_out_channels=(128, 256)  # (256, 512)
    # layers_per_block = 3
    
    block_out_channels=(64, 128, 256)  # (256, 512)
    layers_per_block = 2 # (2,2,3)


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






def sample_lambda(alpha=0.4, batch_size=1, device="cuda"):
    """Sample lambda from Beta distribution"""
    lam = np.random.beta(alpha, alpha, size=batch_size)
    lam = torch.tensor(lam, dtype=torch.float32, device=device)
    return lam.view(-1, 1, 1, 1)  # reshape for broadcasting

# Example inside training loop
def latent_mixup(z_clean, z_broken, alpha=0.5):
    """
    x_full: input with all modalities
    x_partial: input with missing modalities
    target_modalities: ground-truth full modalities (reconstruction target)
    """
    # Encode both

    # Sample lambda for each item in batch
    lam = sample_lambda(alpha=alpha, batch_size=z_clean.size(0), device=z_clean.device)

    # Mix latents
    z_mix = lam * z_broken + (1 - lam) * z_clean

    return z_mix



if __name__ == '__main__':

    image_key = args.input_modality
    
    
    # Image
    image_save_root = "./image_result/step1_ae_train_ct/"
    os.makedirs(image_save_root, exist_ok=True)
    
    if isinstance(image_key, (list, tuple)):
        image_key_str = "-".join(image_key)
    else:
        image_key_str = str(image_key)
    if message != "":
        image_key_str += f"-{message}"


    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------- Define Dataloader ----------------
    in_channels = len(image_key)  # 4 channels:
    dimension = 2
    latent_channels = 8  # 4, 8, 16
    spatial_size = (144, 144)    # (256, 256, 64)
    # spatial_size = (256, 256)  # bs = 8

    key_to_load  = []         
    key_to_load.extend(image_key)

    train_loader, train_ds     = create_paired_dataloader(csv_path=args.dataset_csv, resolution=256, split="train", batch_size=args.batch_size, spatial_size=spatial_size)
    test_loader,  test_ds      = create_paired_dataloader(csv_path=args.dataset_csv, resolution=256, split="test",  batch_size=args.batch_size, spatial_size=spatial_size)


    # ---------------- Define AutoEncoder Model ----------------
    autoencoder   = define_2DAE(in_channels=in_channels, 
                                out_channels=in_channels,
                                latent_channels = latent_channels).to(DEVICE).float()


    discriminator = init_patch_discriminator(args.disc_ckpt, 
                                             spatial_dims=dimension,
                                             in_channels=in_channels, 
                                             num_layers_d=3).to(DEVICE)
 


    def remove_module_prefix(state_dict):
        from collections import OrderedDict
        """Remove 'module.' prefix from keys (if exists)."""
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[new_key] = v
        return new_state_dict

    if args.resume:
        dis_path = os.path.join(args.output_dir, f'dis-{args.resume}-{image_key_str}.pth')
        gen_path = os.path.join(args.output_dir, f'ae-{args.resume}-{image_key_str}.pth')
        if os.path.exists(dis_path):
            discriminator.load_state_dict(remove_module_prefix(torch.load(dis_path)))
        else:
            print(f"Discriminator checkpoint not found: {dis_path}. Starting from scratch.")

        if os.path.exists(gen_path):
            autoencoder.load_state_dict(remove_module_prefix(torch.load(gen_path)))
        else:
            print(f"Autoencoder checkpoint not found: {gen_path}. Starting from scratch.")
        print("Resuming from checkpoint:", gen_path)

    use_mask_loss     = False
    use_adv           = True  #not args.use_broken
    use_kl_loss       = True

    adv_weight        = 0.025
    perceptual_weight = 0.1     # if  args.use_broken else 0.0  # 0.1  
    kl_weight         = 1e-7  # 1e-7

    def charbonnier(x, y, eps=1e-3):
        return torch.mean(torch.sqrt((x - y)**2 + eps**2))

    l1_loss_fn  = charbonnier      # L1Loss()
    kl_loss_fn  = KLDivergenceLoss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")  # criterion="hinge"




    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perc_loss_fn = PerceptualLoss(spatial_dims=dimension,
                                      network_type="squeeze",
                                      is_fake_3d=True if dimension == 3 else False,
                                      fake_3d_ratio=0.2).to(DEVICE)



    if accelerator.is_main_process:
        all_sum = sum(p.numel() for p in autoencoder.parameters())
        print(f"\nAll parameters: {all_sum / 1e6:.2f} M")
        trainable_sum = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_sum / 1e6:.2f} M")

    # 3. Build optimizer
    warmup_epochs = -1


    # Not warmup
    for p in autoencoder.parameters():
            p.requires_grad = True
    optimizer_g = torch.optim.AdamW(
        autoencoder.parameters(), lr=args.lr
    )


    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr)

    
    scheduler_g = torch.optim.lr_scheduler.CyclicLR(
        optimizer_g,
        base_lr=args.lr * 0.1,      # minimum LR
        max_lr =args.lr,             # maximum LR
        step_size_up=5000,          # number of steps to go from base_lr → max_lr
        mode="triangular2",         # triangular2 decays amplitude after each cycle
        cycle_momentum=False        # must be False for AdamW
    )

    avgloss = utils_usage.AverageLoss()
    writer  = SummaryWriter()
    total_counter = 0

    # ---------------- Prepare for Training ----------------
    autoencoder, discriminator, optimizer_g, optimizer_d, train_loader, test_loader, adv_loss_fn, perc_loss_fn, \
    autoencoder.decode,  autoencoder.encode, autoencoder.encode_multi = accelerator.prepare(
            autoencoder, discriminator, optimizer_g, optimizer_d, 
            train_loader, test_loader, adv_loss_fn, perc_loss_fn, 
            autoencoder.decode,  autoencoder.encode, autoencoder.encode_multi
    )

    # Test at starter
    if accelerator.is_main_process:
        validate_model(autoencoder, test_loader, DEVICE,
                   image_save_root=None, max_batches=6, step_name="Start",
                   image_key=image_key, image_key_str=image_key_str)

    start_epoch = 0

    if args.resume:
        start_epoch = int(args.resume)


    for epoch in range(start_epoch, args.n_epochs-start_epoch):

        if DEVICE == "cuda":
            print("EPOCH: ", epoch)
            print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"Cached GPU memory:    {torch.cuda.memory_reserved()  / 1024 ** 2:.2f} MB")

        autoencoder.train()
        discriminator.train()

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch}')


        for step, batch in progress_bar:
            optimizer_g.zero_grad(set_to_none=True)

            if args.DEBUG and step >= 5:
                break

            if step > 200:  # 390
                break

            ct     = batch['CT'].to(DEVICE)
            ctc    = batch['CTC'].to(DEVICE)


            if use_standard_norm:
                ct , ct_mean, ct_std    = standard_normalize(ct)
                ctc, ctc_mean, ctc_std  = standard_normalize(ctc)
            

            images           = torch.cat([ct, ctc], dim=1)
            double_images    = torch.cat([images, images, images], dim=0)

            clean_inputs           = [ct, ctc]
            clean_modalities_mask  = [True, True]

            
            broken_inputs = [ct, ctc]
            broken_modalities_mask   = [True, True]
        
            B = ct.shape[0]
            
            alpha = 0.5  # typical MixUp parameter
            lam = torch.distributions.Beta(alpha, alpha).sample((B,))  # one lambda per sample
            r = lam.view(B, 1, 1, 1)  # reshape for broadcasting over images
            r = r.to(DEVICE)

            if np.random.rand() < 0.5:  # 50% chance to keep
                ctc_mix = ctc * r + (1 - r) * ct  # Mix
                ctc_mix = ctc_mix.detach()

                broken_inputs = [ct, ctc_mix]
                broken_modalities_mask   = [True, True]
                 
            else:
                ct_mix = ct * r + (1 - r) * ctc
                ct_mix = ct_mix.detach()
                broken_inputs = [ct_mix, ctc]
                
                # if np.random.rand() < 0.5:
                broken_modalities_mask   = [True, True]

            broken_double_images    = torch.cat([images, torch.cat(broken_inputs, dim=1)], dim=0)
            broken_double_images    = broken_double_images.detach()


            # with autocast(enabled=True):
            with accelerator.autocast():
                # reconstruction, z_mu, z_sigma = autoencoder(input_images)  # Masked
                enc_out = autoencoder.encode_multi(clean_inputs, clean_modalities_mask)       # EncodeOutput, (self, inputs, mask, return_dict=True):
                z_dist  = enc_out.latent_dist                # Normal distribution

                # latent mean & std
                z_mu    = z_dist.mean
                z_sigma = z_dist.std


                enc_out = autoencoder.encode_multi(broken_inputs, broken_modalities_mask)       # EncodeOutput, (self, inputs, mask, return_dict=True):
                broken_z_dist  = enc_out.latent_dist                # Normal distribution
                broken_z_mu    = broken_z_dist.mean
                broken_z_sigma = broken_z_dist.std


                use_sample_training = True   #False


                if use_sample_training:
                    z       = z_dist.sample()
                    b_z     = broken_z_dist.sample()
                else:
                    z       = z_mu
                    b_z     = broken_z_mu

                z_mix = latent_mixup(z, b_z)
                dec_out = autoencoder.decode(z_mix)              # DecoderOutput
                mix_reconstruction = dec_out.sample

                # Decode
                dec_out = autoencoder.decode(z)              # DecoderOutput
                reconstruction = dec_out.sample

                dec_out = autoencoder.decode(b_z)              # DecoderOutput
                broken_reconstruction = dec_out.sample

                reconstruction = torch.cat([reconstruction, broken_reconstruction, mix_reconstruction], dim=0)

            gen_loss = torch.tensor(0).to(reconstruction.device)
            with accelerator.autocast():
                if use_adv:
                    logits_fake = discriminator(reconstruction.contiguous())[-1]
                    
             
            if use_adv:           
                gen_loss = adv_weight * adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)

            if not use_mask_loss:
                rec_loss = l1_loss_fn(reconstruction, double_images)   # * 5

            else:
                rec_loss = l1_loss_fn(reconstruction, broken_double_images) 
                

            kld_loss = kl_weight * ( kl_loss_fn(z_mu, z_sigma) + \
                                     kl_loss_fn(broken_z_mu, broken_z_sigma) )

            B, C, H, W = reconstruction.shape


            # --------------- Perceptural Loss ---------------
            recon_dce1 = reconstruction[:, 0:1].repeat(1,3,1,1)
            recon_dce2 = reconstruction[:, 1:2].repeat(1,3,1,1)

            if not use_mask_loss:
                image_dce1 = double_images[:, 0:1].repeat(1,3,1,1)
                image_dce2 = double_images[:, 1:2].repeat(1,3,1,1)
            else:
                image_dce1 = broken_double_images[:, 0:1].repeat(1,3,1,1)
                image_dce2 = broken_double_images[:, 1:2].repeat(1,3,1,1)

            # Duplicate each channel 3× → [B,3,H,W]

            # Stack along batch dim → [B*2, 3, H, W]
            reconstruction_3ch = torch.cat([recon_dce1,  recon_dce2],  dim=0)
            images_3ch         = torch.cat([image_dce1,  image_dce2],  dim=0)

            per_loss = torch.tensor(0).to(reconstruction.device)

            # Perceptual loss
            per_loss = perceptual_weight * perc_loss_fn(
                reconstruction_3ch.float(),
                images_3ch.float().detach()
            )

            # Latent Alignment, wrap kl in FP32
            # z_mu_ct, z_mu_ctc   = torch.chunk(z_dist.mean, 2, dim=0)
            # z_std_ct, z_std_ctc = torch.chunk(z_dist.std, 2, dim=0)
            

            def kl_gaussians(mu1, std1, mu2, std2, weight=1e-4, eps=1e-6):
                """
                KL divergence KL(q1||q2) between two diagonal Gaussians.
                Numerically stable and safe.

                Args:
                    mu1, std1: mean and std of q1, shape (B, D, ...)
                    mu2, std2: mean and std of q2, shape (B, D, ...)
                    weight: scaling factor for this KL term
                    eps: clamp value for numerical stability

                Returns:
                    Scalar KL loss
                """
                # clamp std to avoid log(0) or div by zero
                std1 = std1.clamp(min=eps)
                std2 = std2.clamp(min=eps)

                # log-variances
                logvar1 = (std1 ** 2).clamp(min=eps).log()
                logvar2 = (std2 ** 2).clamp(min=eps).log()

                var1 = logvar1.exp()
                var2 = logvar2.exp()

                # KL(q1 || q2) for diagonal Gaussians
                kl = (
                    (var1 / var2)
                    + (mu2 - mu1).pow(2) / var2
                    - 1
                    + (logvar2 - logvar1)
                )

                # flatten across latent dims, then sum per-sample
                kl = kl.flatten(1).sum(1).mean()

                return weight * kl

            
            loss_latent = kl_gaussians(broken_z_mu.float(), 
                                       broken_z_sigma.float(), 
                                       z_mu.detach().float(), 
                                       z_sigma.detach().float()) + \
                                       F.mse_loss(broken_z_mu, z_mu.detach())
            
            loss_latent = 0.1 * loss_latent
          
            loss_g = rec_loss + kld_loss + gen_loss + per_loss + loss_latent

            progress_bar.set_postfix(loss_g=loss_g.item()         if hasattr(loss_g, "item") else loss_g,
                                        latent=loss_latent.item() if hasattr(loss_latent, "item") else loss_latent,
                                        per_loss=per_loss.item()  if hasattr(per_loss, "item") else per_loss,
                                        rec_loss=rec_loss.item()  if hasattr(rec_loss, "item") else rec_loss,
                                        kld_loss=kld_loss.item()  if hasattr(kld_loss, "item") else kld_loss,
                                        gen_loss=gen_loss.item()  if hasattr(gen_loss, "item") else gen_loss)

            
            optimizer_g.zero_grad(set_to_none=True)

            # Check for NaN in the loss
            try:
                accelerator.backward(loss_g)

            except Exception as e:
                print(f"Error during backward: {e}")
                optimizer_g.zero_grad(set_to_none=True)
                accelerator.wait_for_everyone() 

                gen_path = os.path.join(args.output_dir, f'ae-{epoch}-{image_key_str}.pth')

                if os.path.exists(gen_path):
                    autoencoder.load_state_dict(remove_module_prefix(torch.load(gen_path)))
                else:
                    print(f"Autoencoder checkpoint not found: {gen_path}. Starting from scratch.")
                print("Resuming from checkpoint:", gen_path)

                optimizer_g = torch.optim.AdamW(autoencoder.parameters(), lr=args.lr)
                scheduler_g = torch.optim.lr_scheduler.CyclicLR(
                    optimizer_g,
                    base_lr=args.lr * 0.1,
                    max_lr=args.lr,
                    step_size_up=5000,
                    mode="triangular2",
                    cycle_momentum=False
                )   
                autoencoder, optimizer_g, autoencoder.decode,  autoencoder.encode = accelerator.prepare(
                    autoencoder, optimizer_g, autoencoder.decode,  autoencoder.encode
                )

                continue
   
            optimizer_g.step()
            scheduler_g.step() 
            
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)


            del z_mu, z_sigma, loss_g

            # ⚠️ This is a workaround, but should be improved
            if use_adv:
                with accelerator.autocast():
                    fake_images = reconstruction.detach()  # Detach to cut generator graph
                    logits_real = discriminator(images.contiguous())[-1]   # .contiguous().detach()
                d_loss_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)

                discriminator_loss = (d_loss_real) * 0.5
                loss_d = discriminator_loss

                optimizer_d.zero_grad()
                accelerator.backward(loss_d)

                del logits_real, loss_d
          
                with accelerator.autocast():
                    fake_images = reconstruction.detach()  # Detach to cut generator graph
                    logits_fake = discriminator(fake_images.contiguous())[-1]

                d_loss_fake = adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)

                discriminator_loss = (d_loss_fake) * 0.5
                loss_d1 = discriminator_loss

                accelerator.backward(loss_d1)
                optimizer_d.step()
                optimizer_d.zero_grad()

                del logits_fake, loss_d1, reconstruction, fake_images

                torch.cuda.empty_cache()
         

        if accelerator.is_main_process:

            _image_save_root = f"{image_save_root}/epoch_{epoch}"
            os.makedirs(_image_save_root, exist_ok=True)

            autoencoder.eval()
            validate_model(autoencoder, test_loader, DEVICE,
                        image_save_root=_image_save_root, max_batches=6,
                        step_name="Epoch_{}".format(epoch), image_key=image_key)

            
            # 保存模型
            if (epoch + 1) % save_epoch == 0 and accelerator.is_main_process:
                torch.save(discriminator.state_dict(), os.path.join(args.output_dir,
                                                                    f'dis-{epoch + 1}-{image_key_str}.pth'))
                
                torch.save(autoencoder.state_dict(),
                        os.path.join(args.output_dir, f'ae-{epoch + 1}-{image_key_str}.pth'))

                print("Saving models to: ",
                    os.path.join(args.output_dir, f'ae-{epoch + 1}-{image_key_str}.pth'))

                # try:
                #     os.remove(os.path.join(args.output_dir, f'dis-{epoch + 1 - 3}-{image_key_str}.pth'))
                #     os.remove(os.path.join(args.output_dir, f'ae-{epoch + 1 - 3}-{image_key_str}.pth'))
                # except:
                #     pass


        accelerator.wait_for_everyone() 
        gc.collect()
        torch.cuda.empty_cache()


torch.save(autoencoder.state_dict(), os.path.join(args.output_dir, f'ae-final-{image_key_str}.pth'))
print("Training finished.")


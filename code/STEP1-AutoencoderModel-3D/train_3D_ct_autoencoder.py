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

from src.ct_train_dataloader import get_ct_dataloader
from accelerate import Accelerator

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

use_standard_norm = True   # use_standard_norm
use_broken        = True

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

    avg_psnr, avg_ssim = [], []

    # Print the model architecture, and the input and output channel sizes of each layer
    DEBUG = False
    if DEBUG:
        print("Model architecture:\n")
        print(model)  # Print full model

        print("\nLayer-wise input and output shapes:")
        print_model_shapes(model, input_size=(2, len(image_key), 96, 96, 96))

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):

            if args.DEBUG and idx >= 5:
                break

            if idx >= max_batches:
                break


            # images = torch.cat(batch[image_key], dim=1).to(DEVICE)
            ct           = batch['CT'].to(DEVICE)
            ctc          = batch['CTC'].to(DEVICE)
            images       = torch.cat([ct, ctc], dim=1)
            zero_ctc     = torch.zeros_like(ctc)


            if use_standard_norm:
                ct, ct_mean, ct_std   = standard_normalize(ct)
                ctc, ctc_mean, ctc_std = standard_normalize(ctc)   

            if use_broken:
                input_images = torch.cat([ct, zero_ctc], dim=1)
            else:
                input_images = torch.cat([ct, ctc], dim=1)


            
            with accelerator.autocast():
                reconstruction, z_mu, z_sigma = model(input_images)
                

            if use_standard_norm:
                recon_ct_norm  = reconstruction[:, 0:1]#.unsqueeze(1)   # (B,1,H,W)
                recon_ctc_norm = reconstruction[:, 1:2]#.unsqueeze(1)    # (B,1,H,W)
                
                # Denormalize each branch (dims match because mean/std are (B,1,1,1))
                recon_ct  = denormalize(recon_ct_norm, ct_mean, ct_std)
                recon_ctc = denormalize(recon_ctc_norm, ctc_mean, ctc_std)

                # Final reconcat
                reconstruction = torch.cat([recon_ct, recon_ctc], dim=1)  # (B,2,H,W)


            # Move to CPU for metrics and visualization
            image_np = images.cpu().numpy()
            recon_np = reconstruction.cpu().numpy()


            # Example usage:
            # image_np, recon_np = (2, 2, 96, 96)

            # print("image_np stat = ", image_np.shape, recon_np.shape)


            psnr_val, ssim_val = batch_psnr_ssim(image_np, recon_np, data_range=1.0)


            avg_psnr.append(psnr_val)
            avg_ssim.append(ssim_val)

            # Save middle slice comparison image
     
            # Save middle slice comparison image
            middle_slice = image_np.shape[-1] // 2
            image_mid = image_np[:, :, :, :, middle_slice]  # B, C, H, W
            recon_mid = recon_np[:, :, :, :, middle_slice]
            
            if image_save_root is not None:
                for b in range(image_mid.shape[0]):
                    orig  = image_mid[b]  # [4, H, W]
                    recon = recon_mid[b]  # [4, H, W]

                    # Concatenate orig and recon for each of the 4 slices → [4, 2H, W]
                    combined_slices = [np.concatenate([orig[i], recon[i]], axis=0) for i in range(image_mid.shape[1])]  # each is [H, 2W]

                    combined = np.concatenate(combined_slices, axis=1)  # [4H, 2W]

                    save_path = os.path.join(image_save_root, f"{step_name}_img_{idx}_{b}.jpg")
                    save_image(save_path, combined)

    print(f"{step_name} - AVG_PSNR: {np.mean(avg_psnr):.2f}, AVG_SSIM: {np.mean(avg_ssim):.4f}")


message = ""

missing_modality = args.missing_modality



def define_3D_AE(inchannel=2, latent_channels = 16):

    from monai.networks.nets import AutoencoderKL
    block_out_channels = (256, 512)  # 128
    n_blocks = len(block_out_channels)

    vae = AutoencoderKL(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=in_channels,
        channels=block_out_channels,      # corresponds to your block_out_channels
        num_res_blocks=3,          # corresponds to your layers_per_block
        latent_channels=latent_channels,
        norm_num_groups=32,
        attention_levels=(False,) * n_blocks,  # no attention at either level
    )
    return vae



    


if __name__ == '__main__':


    image_key = args.input_modality

    
    # Image
    image_save_root = "./image_result/step1_ae_train/"
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
    dimension = 3
    # spatial_size = (96, 96, 24)   # (256, 256, 64)
    spatial_size = (64, 64, 24)  

    # spatial_size = (256, 256)  # bs = 8

    key_to_load  = []         
    key_to_load.extend(image_key)


    train_loader     = get_ct_dataloader(args, mode=["train", "test", "val"], batch_size=args.batch_size,
                                            spatial_size=spatial_size, key_to_load=key_to_load, cache_dir=args.cache_dir)
    test_loader      = get_ct_dataloader(args, mode="test",  batch_size=args.batch_size,
                                            spatial_size=spatial_size, key_to_load=key_to_load, cache_dir=args.cache_dir)



    # ---------------- Define AutoEncoder Model ----------------
    autoencoder   = define_3D_AE(in_channels, latent_channels = 16).to(DEVICE).float()


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
            print("Discriminator resuming from checkpoint:", dis_path)
        else:
            print(f"Discriminator checkpoint not found: {dis_path}. Starting from scratch.")

        if os.path.exists(gen_path):
            autoencoder.load_state_dict(remove_module_prefix(torch.load(gen_path)))
            print("Autoencoder resuming from checkpoint:", gen_path)
        else:
            print(f"Autoencoder checkpoint not found: {gen_path}. Starting from scratch.")

        


    use_mask_loss     = False
    use_adv           = True
    use_kl_loss       = True

    adv_weight        = 0.025
    perceptual_weight = 0.1   # 0.01  
    kl_weight         = 1e-7  # 1e-7

    def charbonnier(x, y, eps=1e-3):
        return torch.mean(torch.sqrt(((x - y).float())**2 + eps))

    l1_loss_fn  = charbonnier      # L1Loss()
    kl_loss_fn  = KLDivergenceLoss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")  # criterion="hinge"

    # adv_loss_fn = disable_inplace_activations(adv_loss_fn)
    # adv_loss_fn.activation = torch.nn.LeakyReLU(negative_slope=0.05, inplace=False)  # <- safe

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
    # optimizer_g = torch.optim.AdamW(trainable, lr=args.lr)

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

    start_epoch = 0
    if args.resume:
        start_epoch = int(args.resume)



    avgloss = utils_usage.AverageLoss()
    writer  = SummaryWriter()
    total_counter = 0

    # ---------------- Prepare for Training ----------------
    autoencoder, discriminator, optimizer_g, optimizer_d, train_loader, test_loader, adv_loss_fn, perc_loss_fn, autoencoder.decode,  autoencoder.encode = accelerator.prepare(
            autoencoder, discriminator, optimizer_g, optimizer_d, 
            train_loader, test_loader, adv_loss_fn, perc_loss_fn, 
            autoencoder.decode,  autoencoder.encode
    )

    # Test at starter
    validate_model(autoencoder, test_loader, DEVICE,
                   image_save_root=None, max_batches=6, step_name="Start",
                   image_key=image_key, image_key_str=image_key_str)

    
    for epoch in range(start_epoch, args.n_epochs-start_epoch):

        if DEVICE == "cuda":
            print("EPOCH: ", epoch)
            print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"Cached GPU memory:    {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

        autoencoder.train()
        discriminator.train()

        # print(" len(train_loader)=", len(train_loader))


        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch}')


        for step, batch in progress_bar:
            optimizer_g.zero_grad(set_to_none=True)

            if args.DEBUG and step >= 5:
                break

            if step > 250:
                break

            ct    = batch['CT'].to(DEVICE)
            ctc   = batch['CTC'].to(DEVICE)

            if use_standard_norm:
                ct, ct_mean, ct_std    = standard_normalize(ct)
                ctc, ctc_mean, ctc_std = standard_normalize(ctc)

            # print("CT stat:",  ct.min(), ct.max(), ct.mean())
            # print("CTC stat:", ctc.min(), ctc.max(), ctc.mean())
 
            images = torch.cat([ct, ctc], dim=1)

            B, C, H, W, D = images.shape
            if np.random.rand() < 0.5:
                mask_out = torch.zeros((B, 1, 1, 1, 1), device=images.device)  # shape (B,1,1,1)
                ctc_in = mask_out * ctc.clone()
            else:
                ctc_in = ctc.clone()

            if use_broken:
                input_images = torch.cat([ct, ctc_in], dim=1)
            else:
                input_images = torch.cat([ct, ctc], dim=1)



            # with autocast(enabled=True):
            with accelerator.autocast():
                # reconstruction, z_mu, z_sigma = autoencoder(input_images)  # Masked
                reconstruction, z_mu, z_sigma = autoencoder(input_images)
                
                
                # print("CT stat:",  reconstruction.shape, z_mu.shape)



                if use_adv:
                    logits_fake = discriminator(reconstruction.contiguous())[-1]
                    gen_loss = adv_weight * adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)
                else:
                    gen_loss = torch.tensor(0).to(reconstruction.device)


                if not use_mask_loss:
                    rec_loss = l1_loss_fn(reconstruction, images)   # * 5

                else:
                    mask = mask.expand_as(images)
                    rec_loss = torch.abs(reconstruction * mask - images * mask).sum() / torch.sum(mask)
                
                z_sigma = torch.clamp(z_sigma, min=1e-6, max=1e2)
                kld_loss = kl_weight * kl_loss_fn(z_mu, z_sigma)
            
                B, C, H, W, D = reconstruction.shape

                # --------------- Perceptural Loss ---------------
                # Split into CT and CTC
                recon_ct,  recon_ctc  = reconstruction[:,0:1], reconstruction[:,1:2]  # [B,1,H,W]
                image_ct,  image_ctc  = images[:,0:1],         images[:,1:2]

                # Duplicate each channel 3× → [B,3,H,W]
                recon_ct_3  = recon_ct.repeat(1,3,1,1,1)
                recon_ctc_3 = recon_ctc.repeat(1,3,1,1,1)
                image_ct_3  = image_ct.repeat(1,3,1,1,1)
                image_ctc_3 = image_ctc.repeat(1,3,1,1,1)
                
                if use_standard_norm:
                    recon_ct_3  = denormalize(recon_ct_3,  ct_mean,  ct_std)
                    recon_ctc_3 = denormalize(recon_ctc_3, ctc_mean, ctc_std)
                    image_ct_3  = denormalize(image_ct_3,  ct_mean,  ct_std)
                    image_ctc_3 = denormalize(image_ctc_3, ctc_mean, ctc_std)



                # Stack along batch dim → [B*2, 3, H, W]
                reconstruction_3ch = torch.cat([recon_ct_3,  recon_ctc_3],  dim=0)
                images_3ch         = torch.cat([image_ct_3,  image_ctc_3],  dim=0)

                per_loss = torch.tensor(0).to(reconstruction.device)

                # Perceptual loss
                per_loss = perceptual_weight * perc_loss_fn(
                    reconstruction_3ch.float(),
                    images_3ch.float().detach()
                )


                loss_g = rec_loss + kld_loss + gen_loss + per_loss

                progress_bar.set_postfix(loss_g=loss_g.item() if hasattr(loss_g, "item") else loss_g,
                                         per_loss=per_loss.item() if hasattr(loss_g, "item") else per_loss,
                                         rec_loss=rec_loss.item() if hasattr(rec_loss, "item") else rec_loss,
                                         kld_loss=kld_loss.item() if hasattr(kld_loss, "item") else kld_loss,
                                         gen_loss=gen_loss.item() if hasattr(gen_loss, "item") else gen_loss)

            
            optimizer_g.zero_grad(set_to_none=True)
            accelerator.backward(loss_g)
   
            optimizer_g.step()
            scheduler_g.step() 
            

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

                # optimizer_d.zero_grad()
                accelerator.backward(loss_d1)
                optimizer_d.step()
                optimizer_d.zero_grad()

                del logits_fake, loss_d1, reconstruction, fake_images

                torch.cuda.empty_cache()
         



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

            try:
                os.remove(os.path.join(args.output_dir, f'dis-{epoch + 1 - 3}-{image_key_str}.pth'))
                os.remove(os.path.join(args.output_dir, f'ae-{epoch + 1 - 3}-{image_key_str}.pth'))
            except:
                pass

        if epoch + 1 == warmup_epochs:
            print("🔓 Unfreezing full model...")
            for p in autoencoder.parameters():
                p.requires_grad = True
            optimizer_g = torch.optim.AdamW(
                autoencoder.parameters(), lr=args.lr
            )
            optimizer_g = accelerator.prepare(optimizer_g)

        # accelerator.wait_for_everyone() 
        gc.collect()
        torch.cuda.empty_cache()


torch.save(autoencoder.state_dict(), os.path.join(args.output_dir, f'ae-final-{image_key_str}.pth'))
print("Training finished.")


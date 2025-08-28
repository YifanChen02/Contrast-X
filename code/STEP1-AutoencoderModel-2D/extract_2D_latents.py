import os, gc, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils import args
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
from pathlib import Path

import warnings
import numpy as np
import torch
from tqdm import tqdm
from monai.utils import set_determinism

import matplotlib.pyplot as plt

from utils import args, import_from_dotted_path, utils_metric
from src import utils_usage


from src import init_autoencoder, KLDivergenceLoss, init_patch_discriminator, GradientAccumulation
from utils.utils_image import save_image, pad_to_shape

from src.brats_dataloader import get_brats_dataloader
from accelerate import Accelerator
import shutil

accelerator = Accelerator()
DEVICE = accelerator.device


from collections import OrderedDict


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




def define_2DAE(inchannel=3, latent_channels = 16):
    from huggingface_hub import snapshot_download
    from diffusers import AutoencoderKL

    

    # local_dir = snapshot_download("stabilityai/sd-vae-ft-ema")  

    
    from diffusers.models import AutoencoderKL
    in_channels = 2
    
    vae = AutoencoderKL(
        in_channels=in_channels, out_channels=in_channels, latent_channels=latent_channels,
        block_out_channels=(128, 256),  # (128, 256),(256, 512) # 2 blocks -> ×4 total downsample
        down_block_types=("DownEncoderBlock2D","DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D","UpDecoderBlock2D"),
        layers_per_block=3, 
        norm_num_groups=32
    )

    return vae


def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = v
    return new_state_dict

def extract_middle_slices_per_channel(volume):
    # volume: (C, H, W, D)
    C, H, W, D = volume.shape
    axial_slices = []
    coronal_slices = []
    sagittal_slices = []

    for c in range(C):
        vol_c = volume[c]
        axial    = vol_c[:, :, D // 2]
        coronal  = vol_c[:, W // 2, :]
        sagittal = vol_c[H // 2, :, :]
        axial_slices.append(axial)
        coronal_slices.append(coronal)
        sagittal_slices.append(sagittal)

    return axial_slices, coronal_slices, sagittal_slices


if __name__ == '__main__':

    image_key = args.input_modality

    if isinstance(image_key, (list, tuple)):
        image_key_str = "-".join(image_key)
    else:
        image_key_str = str(image_key)

    DATA_ROOT = Path(args.data_dir)
    OUT_ROOT  = Path(args.output_dir)  # wherever you want latents

    in_channels = len(image_key)  # 4 channels: t1c, t1n, t2w, t2f
    dimension = 3
    spatial_size = (256, 256)  

    key_to_load = ["mask"]  # ["mask", "density"]
    # image_key = ["t1c", "t1n", "t2w", "t2f"]
    key_to_load.extend(image_key)

    from src.ct_2D_dataloader import create_paired_dataloader
    data_loader, train_ds     = create_paired_dataloader(csv_path=args.dataset_csv, split="test_all", batch_size=args.batch_size, spatial_size=spatial_size)


    print("Setting up Autoencoder model...")
    autoencoder   = define_2DAE(in_channels, latent_channels = 16).float()    # .to(DEVICE).float()


    print("Finish setting up...")
    weight = torch.load(args.aekl_ckpt)
    weight = remove_module_prefix(weight)

    autoencoder.load_state_dict(weight)

    autoencoder.to(DEVICE)
    autoencoder.eval()  # Important for inference

    autoencoder, data_loader, autoencoder.encode, autoencoder.decode = accelerator.prepare(autoencoder, data_loader, autoencoder.encode, autoencoder.decode)
    

    # shutil.rmtree(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, image_key_str)

    with torch.no_grad(), accelerator.autocast():
    # with torch.no_grad():
        os.makedirs(args.output_dir, exist_ok=True)  # 确保输出目录存在
        ids = 0

        print("Test Source...")
        for batch in tqdm(data_loader, total=len(data_loader)):
            ct           = batch['CT'].to(DEVICE)
            ctc          = batch['CTC'].to(DEVICE)
            if use_standard_norm:
                ct, ct_mean, ct_std   = standard_normalize(ct)
                ctc, ctc_mean, ctc_std = standard_normalize(ctc)   

            images       = torch.cat([ct, ctc], dim=1)

            patient_id   = batch["CT_path"]  # 获取源文件路径

            images = images.to(DEVICE)

            enc_out = autoencoder.encode(images)   # EncodeOutput
            z_dist  = enc_out.latent_dist                # Normal distribution
            z_mu    = z_dist.mean

            mri_latent = z_mu.cpu().numpy() 


            print("mri_latent = ", mri_latent.shape)   # (16, 16, 128, 128)
            print("patient_id = ", patient_id[0] )     # 
            # /date/hao/PairedContrast/CT/low_256x256_2Dimension/Stomach_Colon_Liver_Pancreas_CT_train_val_test/train/HCC_023_1999-12-24_CT/slice_033/CT.jpg

        

            # 保存潜在表示, in Batch
            for i, latent in enumerate(mri_latent):
                img_path = Path(patient_id[i])  # e.g. /.../Stomach_.../train/HCC_023_1999-12-24_CT/slice_033/CT.jpg

                # ---- get "patient_path" you asked for ----
                rel_dir = img_path.relative_to(DATA_ROOT).parent
                patient_path = str(rel_dir)  # "Stomach_Colon_Liver_Pancreas_CT_train_val_test/train/HCC_023_1999-12-24_CT/slice_033"

                # ---- make an output dir that mirrors the dataset tree ----
                out_dir = OUT_ROOT / rel_dir
                out_dir.mkdir(parents=True, exist_ok=True)

                destpath = out_dir / 'CT_CTC.npz'
                np.savez_compressed(destpath, data=latent)
                # print("destpath = ", destpath)



            # ct           = batch['CT'].to(DEVICE)
            ctc          = torch.zeros_like(batch['CTC']).to(DEVICE)
            images       = torch.cat([ct, ctc], dim=1)
            images = images.to(DEVICE)


            enc_out    = autoencoder.encode(images)   # EncodeOutput
            z_dist     = enc_out.latent_dist                # Normal distribution
            z_mu       = z_dist.mean
            mri_latent = z_mu.cpu().numpy() 

            for i, latent in enumerate(mri_latent):
                img_path = Path(patient_id[i])  # e.g. /.../Stomach_.../train/HCC_023_1999-12-24_CT/slice_033/CT.jpg

                # ---- get "patient_path" you asked for ----
                rel_dir = img_path.relative_to(DATA_ROOT).parent
                patient_path = str(rel_dir)  # "Stomach_Colon_Liver_Pancreas_CT_train_val_test/train/HCC_023_1999-12-24_CT/slice_033"

                # ---- make an output dir that mirrors the dataset tree ----
                out_dir = OUT_ROOT / rel_dir
                out_dir.mkdir(parents=True, exist_ok=True)

                destpath = out_dir / 'CT.npz'
                np.savez_compressed(destpath, data=latent)


            del mri_latent
            gc.collect()
            torch.cuda.empty_cache()

            if ids == 1:
                # Decode
                dec_out = autoencoder.decode(z_mu)              # DecoderOutput
                reconstruction = dec_out.sample
                
                # reconstruction, z_mu, z_sigma = autoencoder(images.float())
                os.makedirs("./reconstruction", exist_ok=True)

                reconstruction = reconstruction.cpu().numpy()

                modalities = args.input_modality  # ["T1c", "T1n", "T2w", "T2f"]
                # planes = ["Axial", "Coronal", "Sagittal"]

                # 保存重建图像
                
                def as_CHW(x):
                    """Return np array shaped (C, H, W) from (H,W) or (C,H,W)."""
                    if isinstance(x, torch.Tensor):
                        x = x.detach().cpu().numpy()
                    x = np.asarray(x).squeeze()
                    if x.ndim == 2:          # (H, W) -> add channel dim
                        x = x[None, ...]
                    elif x.ndim == 3:        # (C, H, W) -> ok
                        pass
                    else:
                        raise ValueError(f"2D only: got shape {x.shape}")
                    x = np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)
                    return x

                def sanitize_filename(s: str) -> str:
                    # Replace path separators and unsafe chars so we can save as a single file
                    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", s.replace(os.sep, "__"))

                out_dir = Path("./reconstruction")
                out_dir.mkdir(parents=True, exist_ok=True)

                # how many rows (IDs) to show
                B = min(len(reconstruction), 5)

                # figure layout from the first sample
                first = as_CHW(reconstruction[0])
                C = first.shape[0]
                use_two_modalities = C >= 2
                ncols = 4 if use_two_modalities else 2
                col_titles = ["CT", "CT_recon"] if not use_two_modalities else ["CT", "CT_recon", "CTC", "CTC_recon"]

                fig, axs = plt.subplots(B, ncols, figsize=(3*ncols, 3*B))
                axs = np.atleast_2d(axs)

                # column headers
                for c, t in enumerate(col_titles):
                    axs[0, c].set_title(t)

                for i in range(B):
                    rec_i = as_CHW(reconstruction[i])  # (C,H,W)
                    img_i = as_CHW(images[i])          # (C,H,W)
                    if rec_i.shape != img_i.shape:
                        raise ValueError(f"Sample {i}: recon{rec_i.shape} vs img{img_i.shape}")

                    # row label = patient id (last path part)
                    rid = patient_id[i] if i < len(patient_id) else f"id_{i}"
                    axs[i, 0].set_ylabel(Path(str(rid)).name, rotation=90, fontsize=8)

                    # CT (channel 0)
                    axs[i, 0].imshow(img_i[0], cmap="gray", vmin=0, vmax=1)   # CT
                    axs[i, 0].axis("off")
                    axs[i, 1].imshow(rec_i[0], cmap="gray", vmin=0, vmax=1)   # CT_recon
                    axs[i, 1].axis("off")

                    if use_two_modalities:
                        # CTC (channel 1)
                        axs[i, 2].imshow(img_i[1], cmap="gray", vmin=0, vmax=1)  # CTC
                        axs[i, 2].axis("off")
                        axs[i, 3].imshow(rec_i[1], cmap="gray", vmin=0, vmax=1)  # CTC_recon
                        axs[i, 3].axis("off")

                plt.tight_layout()
                save_path = out_dir / "batch_grid.jpg"
                plt.savefig(save_path, dpi=200, bbox_inches="tight")
                plt.close()
                print("Saved grid to", save_path)

            ids += 1


    print("Latent representations saved successfully.")



from diffusers.models import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKLOutput, DiagonalGaussianDistribution, DecoderOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
import torch
from torch import nn

# ------------------------------
# Utility: Product-of-Experts fusion
# ------------------------------
def poe(mu_list, logvar_list):
    var_list = [torch.exp(lv) for lv in logvar_list]
    prec_list = [1.0 / v for v in var_list]

    prec_sum = torch.stack(prec_list).sum(dim=0)
    mu = torch.stack([m * p for m, p in zip(mu_list, prec_list)]).sum(dim=0) / prec_sum
    var = 1.0 / prec_sum
    logvar = torch.log(var)
    return mu, logvar

def max_abs_fusion(mu_list, logvar_list):
    """
    Fuse latents by selecting, for each dimension,
    the mean with the largest absolute value (distinct signal).
    Corresponding logvar is carried along from the same expert.
    """
    # stack into tensors of shape (num_experts, latent_dim, ...)
    mu_stack = torch.stack(mu_list)          # [N, ...]
    logvar_stack = torch.stack(logvar_list)  # [N, ...]

    # index of max-abs value along expert dimension
    max_idx = mu_stack.abs().argmax(dim=0)   # shape like latent_dim

    # gather selected means and logvars
    mu = torch.gather(mu_stack, 0, max_idx.unsqueeze(0)).squeeze(0)
    logvar = torch.gather(logvar_stack, 0, max_idx.unsqueeze(0)).squeeze(0)

    return mu, logvar

# ------------------------------
# Utility: Mixture-of-Experts fusion
# ------------------------------
def moe(mu_list, logvar_list):
    var_list = [torch.exp(lv) for lv in logvar_list]
    mu_stack = torch.stack(mu_list)
    var_stack = torch.stack(var_list)

    # Uniform weights
    weights = torch.ones(len(mu_list), device=mu_stack.device) / len(mu_list)

    # Weighted mean
    mu = torch.sum(weights[:, None] * mu_stack, dim=0)

    # Total variance = E[var] + Var[mean]
    var_exp = torch.sum(weights[:, None] * var_stack, dim=0)
    var_mean = torch.sum(weights[:, None] * (mu_stack - mu)**2, dim=0)
    var = var_exp + var_mean

    logvar = torch.log(var + 1e-8)
    return mu, logvar







class AutoencoderKL_multi_encoder(AutoencoderKL):
    def __init__(self, num_encode=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.num_encode = num_encode

        # Duplicate encoders
        self.encoders = nn.ModuleList([self._build_encoder_copy() for _ in range(num_encode)])
        self.quant_convs = nn.ModuleList(
            [self._build_conv_copy(self.quant_conv) for _ in range(num_encode)]
        )
        self.post_quant_convs = nn.ModuleList(
            [self._build_conv_copy(self.post_quant_conv) for _ in range(num_encode)]
        )


    def _build_encoder_copy(self):
        """Deep-copy the encoder so each is independent."""
        import copy
        return copy.deepcopy(self.encoder)

    def _build_conv_copy(self, conv):
        """Clone quant_conv or post_quant_conv safely."""
        import copy
        return copy.deepcopy(conv) if conv is not None else None


        
    def _encode(self, x: torch.Tensor, encoder_id: int = 0) -> torch.Tensor:
        batch_size, num_channels, height, width = x.shape

        if self.use_tiling and (width > self.tile_sample_min_size or height > self.tile_sample_min_size):
            return self._tiled_encode(x)

        enc = self.encoders[encoder_id](x)

        quant_conv = self.quant_convs[encoder_id]
        if quant_conv is not None:
            enc = quant_conv(enc)

        return enc

    @apply_forward_hook
    def encode(self, x: torch.Tensor, encoder_id: int = 0, return_dict: bool = True):
        """
        Encode using one encoder (like vanilla AutoencoderKL).
        """
        if encoder_id >= self.num_encode:
            raise ValueError(f"encoder_id {encoder_id} out of range (num_encode={self.num_encode})")

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice, encoder_id) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x, encoder_id)

        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    # ------------------------------
    # Multi-encoder fusion with PoE
    # ------------------------------
    def encode_multi(self, inputs, mask, return_dict=True):
        """
        inputs: list of modality tensors
        mask: list of booleans (True if modality present)
        """
        mu_list, logvar_list = [], []
        for i, present in enumerate(mask):
            if present:
                h = self._encode(inputs[i], encoder_id=i)
                posterior = DiagonalGaussianDistribution(h)
                mu_list.append(posterior.mean)
                logvar_list.append(posterior.logvar)

        if len(mu_list) == 0:
            batch = inputs[0].shape[0]
            device = inputs[0].device
            mu = torch.zeros(batch, self.latent_dim, device=device)
            logvar = torch.zeros_like(mu)
        else:
            # mu, logvar = poe(mu_list, logvar_list)
            mu, logvar = max_abs_fusion(mu_list, logvar_list)

        new_params = torch.cat([mu, logvar], dim=1)
        fused_posterior = DiagonalGaussianDistribution(new_params)

        # DiagonalGaussianDistribution.from_mean_logvar(mu, logvar)

        if not return_dict:
            return (fused_posterior,)
        return AutoencoderKLOutput(latent_dist=fused_posterior)




    @apply_forward_hook
    def decode(
        self, z: torch.FloatTensor, return_dict: bool = True, generator=None
    ):
        """
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)
    
    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        use_multiple_encoders: bool = False,
        modalities_mask = None,
        generator = None,
    ):
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        if use_multiple_encoders:
            if not isinstance(sample, (list, tuple)):
                raise ValueError("When use_multiple_encoders=True, `sample` must be a list of tensors.")
            if modalities_mask is None:
                modalities_mask = [True] * len(sample)
            posterior = self.encode_multi(sample, mask=modalities_mask, return_dict=True).latent_dist
        else:
            posterior = self.encode(sample, return_dict=True).latent_dist


        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

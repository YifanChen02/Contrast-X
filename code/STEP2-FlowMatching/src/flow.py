import torch


def pad_v_like_x(v_, x_):
    """
    Function to reshape the vector by the number of dimensions
    of x. E.g. x (bs, c, h, w), v (bs) -> v (bs, 1, 1, 1).
    """
    if isinstance(v_, float):
        return v_
    return v_.reshape(-1, *([1] * (x_.ndim - 1)))

# From: https://colab.research.google.com/github/atong01/conditional-flow-matching/blob/main/examples/2D_tutorials/Flow_matching_tutorial.ipynb#scrollTo=278f89ee
class LinearSchedule:
    def alpha_t(self, t):
        return t
    
    def alpha_dt_t(self, t):
        return 1
    
    def sigma_t(self, t):
        return 1 - t
    
    def sigma_dt_t(self, t):
        return -1

    """ Legacy functions to work with SiT Sampler """

    def compute_alpha_t(self, t):
        return self.alpha_t(t), self.alpha_dt_t(t)
    
    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        return self.sigma_t(t), self.sigma_dt_t(t)
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        """Compute the ratio between d_alpha and alpha"""
        return 1 / t
    
    def compute_drift(self, x, t):
        """We always output sde according to score parametrization; """
        t = pad_v_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t

        return -drift, diffusion
    
    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        """Compute the diffusion term of the SDE
        Args:
          x: [batch_dim, ...], data point
          t: [batch_dim,], time vector
          form: str, form of the diffusion term
          norm: float, norm of the diffusion term
        """
        t = pad_v_like_x(t, x)
        choices = {
            "constant": norm,
            "SBDM": norm * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * torch.cos(np.pi * t) + 1) ** 2,
            "increasing-decreasing": norm * torch.sin(np.pi * t) ** 2,
        }

        try: diffusion = choices[form]
        except KeyError: raise NotImplementedError(f"Diffusion form {form} not implemented")
        
        return diffusion
    

    
    def get_score_from_velocity(self, velocity, x, t):
        """Wrapper function: transfrom velocity prediction model to score
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = pad_v_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var
        return score
    
    def get_noise_from_velocity(self, velocity, x, t):
        """Wrapper function: transfrom velocity prediction model to denoiser
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = pad_v_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = reverse_alpha_ratio * d_sigma_t - sigma_t
        noise = (reverse_alpha_ratio * velocity - mean) / var
        return noise

    def get_velocity_from_score(self, score, x, t):
        """Wrapper function: transfrom score prediction model to velocity
        Args:
            score: [batch_dim, ...] shaped tensor; score model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = pad_v_like_x(t, x)
        drift, var = self.compute_drift(x, t)
        velocity = var * score - drift
        return velocity
    

class GVPSchedule(LinearSchedule):
    def alpha_t(self, t):
        return torch.sin(t * math.pi / 2)
    
    def alpha_dt_t(self, t):
        return 0.5 * math.pi * torch.cos(t * math.pi / 2)
    
    def sigma_t(self, t):
        return torch.cos(t * math.pi / 2)
    
    def sigma_dt_t(self, t):
        return - 0.5 * math.pi * torch.sin(t * math.pi / 2)
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        """Special purposed function for computing numerical stabled d_alpha_t / alpha_t"""
        return np.pi / (2 * torch.tan(t * np.pi / 2))



schedule = LinearSchedule()

# elif schedule == "gvp":
# assert sigma_min == 0.0, "GVP schedule does not support sigma_min."
# schedule = GVPSchedule()


"""
Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

Parameters
----------
x0 : Tensor, shape (bs, *dim)
    represents the source minibatch
x1 : Tensor, shape (bs, *dim)
    represents the target minibatch
t : FloatTensor, shape (bs)

Returns
-------
xt : Tensor, shape (bs, *dim)

References
----------
[1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
"""
# def compute_xt(x0, x1, t, sigma_min=0.1):
#     """
#     Sample from the time-dependent density p_t
#         xt ~ N(alpha_t * x1 + sigma_t * x0, sigma_min * I),
#     according to Eq. (1) in [3] and for the linear schedule Eq. (14) in [2].

#     Args:
#         x0 : shape (bs, *dim), represents the source minibatch (noise)
#         x1 : shape (bs, *dim), represents the target minibatch (data)
#         t  : shape (bs,) represents the time in [0, 1]
#     Returns:
#         xt : shape (bs, *dim), sampled point along the time-dependent density p_t
#     """
#     t = pad_v_like_x(t, x0)
#     alpha_t = schedule.alpha_t(t)
#     sigma_t = schedule.sigma_t(t)
#     xt = alpha_t * x1 + sigma_t * x0
#     if sigma_min > 0:
#         xt += sigma_min * torch.randn_like(xt)
#     return xt


# def compute_ut(x0, x1, t):
#     """
#     Compute the time-dependent conditional vector field
#         ut = alpha_dt_t * x1 + sigma_dt_t * x0,
#     see Eq. (7) in [3].

#     Args:
#         x0 : Tensor, shape (bs, *dim), represents the source minibatch (noise)
#         x1 : Tensor, shape (bs, *dim), represents the target minibatch (data)
#         t  : FloatTensor, shape (bs,) represents the time in [0, 1]
#     Returns:
#         ut : conditional vector field
#     """
#     t = pad_v_like_x(t, x0)
#     alpha_dt_t = schedule.alpha_dt_t(t)
#     sigma_dt_t = schedule.sigma_dt_t(t)
#     return alpha_dt_t * x1 + sigma_dt_t * x0



def compute_xt(x0, x1, t, sigma_min=0.1):
    """
    Stochastic interpolation (rectified flow):
    x_t = (1 - t) * x0 + t * x1 + sigma(t) * eps
    """
    t = pad_v_like_x(t, x0)
    sigma_t = sigma_min * torch.sqrt(t * (1 - t))   # schedule
    eps = torch.randn_like(x0)
    xt = (1 - t) * x0 + t * x1 + sigma_t * eps
    return xt, eps, sigma_t


def compute_ut(x0, x1, t, eps, sigma_t, sigma_min=0.1):
    """
    Velocity field for rectified flow:
    u_t = (x1 - x0) + sigma'(t) * eps
    """
    t = pad_v_like_x(t, x0)

    # sigma(t) = sigma_min * sqrt(t * (1 - t))
    sigma_dt = sigma_min * (0.5 - t) / torch.sqrt(t * (1 - t) + 1e-8)

    ut = (x1 - x0) + sigma_dt * eps
    return ut




# t = torch.rand(x0.shape[0]).type_as(x0)
# xt = sample_conditional_pt(x0, x1, t, sigma=0.01)
# ut = compute_conditional_vector_field(x0, x1)

# vt = model(torch.cat([xt, t[:, None]], dim=-1))
# loss = torch.mean((vt - ut) ** 2)






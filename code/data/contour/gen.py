import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Grid for plotting
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x, y)

# Anisotropic Gaussian with perturbation
def irregular_gaussian(x, y, x0, y0, sx=1.0, sy=1.0, angle=0.0, noise_strength=0.2):
    # Rotate coordinates
    xp = np.cos(angle) * (x - x0) + np.sin(angle) * (y - y0)
    yp = -np.sin(angle) * (x - x0) + np.cos(angle) * (y - y0)
    g = np.exp(-(xp**2 / (2*sx**2) + yp**2 / (2*sy**2)))

    # Add smooth random field
    rng = np.random.default_rng(42)  # fixed seed
    noise = rng.normal(size=x.shape)
    smooth_noise = gaussian_filter(noise, sigma=20)  # smooth it
    smooth_noise = (smooth_noise - smooth_noise.min()) / (smooth_noise.max() - smooth_noise.min())
    g = g * (1 + noise_strength * (smooth_noise - 0.5))
    return g

# Define two irregular blobs
Z1 = irregular_gaussian(X, Y, -5, -3, sx=1.5, sy=0.9, angle=0.5, noise_strength=10.0)
Z2 = irregular_gaussian(X, Y,  3.5,  3.5, sx=1.0, sy=1.8, angle=-0.3, noise_strength=1.25)

plt.figure(figsize=(12,12))

z1_ratio = 0.15
Z1_masked = np.ma.masked_less_equal(Z1, z1_ratio)
Z2_masked = np.ma.masked_less_equal(Z2, 0.05)
# Mask values below threshold so background stays transparent

# Solid contour lines
levels1 = np.linspace(z1_ratio, Z1_masked.max(), 8)  # same thresholds for lines & fill
levels2 = np.linspace(0.05, Z2.max(), 8)

contours1 = plt.contour(X, Y, Z1, levels=levels1, colors="#3A4CC0", linewidths=2)
contours2 = plt.contour(X, Y, Z2, levels=levels2, colors="#C03A3A", linewidths=2)

plt.contourf(X, Y, Z1_masked, levels=levels1, cmap="Blues")
plt.contourf(X, Y, Z2_masked, levels=levels2, cmap="Reds")





plt.axis("equal")
plt.axis("off")

# Save with transparent background
plt.savefig("contour_plot.png", bbox_inches="tight", dpi=300, transparent=True)

plt.show()

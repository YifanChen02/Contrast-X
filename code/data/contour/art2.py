import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Grid
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)

# Create smooth random field (noise + Gaussian smoothing)
np.random.seed(42)  # reproducible randomness
noise1 = np.random.rand(*X.shape)
noise2 = np.random.rand(*X.shape)

# Smooth them (like terrain)
Z1 = gaussian_filter(noise1, sigma=15)
Z2 = gaussian_filter(noise2, sigma=15)

plt.figure(figsize=(6,6))

# Contours (random blob shapes, not circles!)
cont1 = plt.contour(X, Y, Z1, levels=8, cmap="Blues", linewidths=2)
cont2 = plt.contour(X, Y, Z2, levels=8, cmap="Reds", linewidths=2)

plt.axis("equal")
plt.axis("off")
plt.savefig("random_contour.png", dpi=300, bbox_inches="tight")
plt.show()

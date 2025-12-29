import numpy as np
import matplotlib.pyplot as plt

# Grid
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)

# Anisotropic Gaussian with angle
def gaussian_aniso(x, y, x0, y0, sigma_x=1.0, sigma_y=1.0, theta=0.0):
    x_shift = x - x0
    y_shift = y - y0
    x_rot = np.cos(theta) * x_shift + np.sin(theta) * y_shift
    y_rot = -np.sin(theta) * x_shift + np.cos(theta) * y_shift
    return np.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2)))

# Two randomish blobs
Z1 = (gaussian_aniso(X, Y, -2, -2, sigma_x=1.5, sigma_y=0.8, theta=np.pi/6)
    + 0.3 * gaussian_aniso(X, Y, -1, -3, sigma_x=0.7, sigma_y=1.2, theta=np.pi/3))

Z2 = (gaussian_aniso(X, Y,  2,  2, sigma_x=1.0, sigma_y=1.3, theta=-np.pi/5)
    + 0.4 * gaussian_aniso(X, Y,  3,  1, sigma_x=0.9, sigma_y=0.6, theta=np.pi/4))

# Add a little smooth noise so contours aren’t too symmetric
rng = np.random.default_rng(42)
Z1 += 0.05 * rng.standard_normal(Z1.shape)
Z2 += 0.05 * rng.standard_normal(Z2.shape)

# Plot
plt.figure(figsize=(6,6))
plt.contourf(X, Y, Z1, levels=15, cmap="Blues", alpha=0.7)
plt.contourf(X, Y, Z2, levels=15, cmap="Reds", alpha=0.7)

plt.contour(X, Y, Z1, levels=6, colors="blue", alpha=0.6, linewidths=1.5)
plt.contour(X, Y, Z2, levels=6, colors="red", alpha=0.6, linewidths=1.5)

plt.axis("equal")
plt.axis("off")
plt.savefig("contour_random.png", dpi=150)
plt.show()

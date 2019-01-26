# In this example, we will show how to use filter to detect edges

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# load the famous Lena image
img = mpimg.imread("lena.png")

print("color lena")
plt.imshow(img)
plt.show()

print()
print("type of img", type(img))
print("shape of img", img.shape)
print()

# Make it black and white
# Get average for three channels
bw = img.mean(axis=2)
print("shape of bw", bw.shape)
print()

print("black lena")
plt.imshow(bw)
plt.show()

# Sobel filter - approximate gradient at each point of image in X direction
# Vertical edge filter
Hx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
], dtype=np.float32)

print()
print("detect vertical edges")
print()
Gx = convolve2d(bw, Hx)

print("Shape of Gx", Gx.shape)
plt.print()
plt.imshow(Gx, cmap="gray")
plt.show()

# Sobel operator - approximate gradient in Y direction
# Horizontal edge detection
Hy = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)

Gy = convolve2d(bw, Hy)
print()
print("shape of Gy", Gy.shape)
print()
plt.imshow(Gy, cmap="gray")
plt.show()

print()
print("Calculate the magnitude of the gradient")
print()
G = np.sqrt(Gx*Gx + Gy*Gy)  # Magnitude
print()
plt.show(G, cmap="gray")
plt.show()

# Plot the gradient direction
print()
print("The gradient direction")
print()
theta = np.arctan2(Gy, Gx)
plt.imshow(theta, cmap="gray")
plt.show()
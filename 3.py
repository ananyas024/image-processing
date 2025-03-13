

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
k = cv2.imread("image.png")
# Display images
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(k,cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')
k = cv2.cvtColor(k, cv2.COLOR_BGR2GRAY)
k1 = np.float64(k)
p_msk = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

# Apply Sobel filter (dx)
kx = convolve2d(k1, p_msk, mode='same', boundary='symm')
# Apply Sobel filter (dy)
ky = convolve2d(k1, p_msk.T, mode='same', boundary='symm')
# Combine the two directional gradients
ked = np.sqrt(kx**2 + ky**2)

plt.subplot(2, 2, 2)
plt.imshow(np.abs(ked), cmap='gray')
plt.title("Sobel Operator")
plt.axis('off')


# Apply another filter (Scharr Filter)
s_msk = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
# Apply Scharr filter (dx)
kx = convolve2d(k1, s_msk, mode='same', boundary='symm')
# Apply Scharr filter (dy)
ky = convolve2d(k1, s_msk.T, mode='same', boundary='symm')
# Combine the gradients
ked = np.sqrt(kx**2 + ky**2)

# Display images again
plt.subplot(2, 2, 3)
plt.imshow(np.abs(ked), cmap='gray')
plt.title("Scharr Operator")
plt.axis('off')

# Define Prewitt filter
p_msk = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
# Apply Prewitt filter (dx)
kx_prewitt = convolve2d(k1, p_msk, mode='same', boundary='symm')
# Apply Prewitt filter (dy)
ky_prewitt = convolve2d(k1, p_msk.T, mode='same', boundary='symm')
# Combine the gradients for Prewitt
ked_prewitt = np.sqrt(kx_prewitt**2 + ky_prewitt**2)
# Display images for Prewitt Edge Detection
plt.subplot(2, 2, 4)
plt.imshow(np.abs(ked_prewitt), cmap='gray')
plt.title("Prewitt Operator")
plt.axis('off')
plt.show()
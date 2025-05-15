import matplotlib.pyplot as plt
import numpy as np

# Image dimensions
size = 20

# Clean Images (Ground Truth)
clean_img1 = np.zeros((size, size))
clean_img2 = np.zeros((size, size))

# Ground Truth Poisoned Masked Images
poisoned_img1 = clean_img1.copy()
poisoned_img2 = clean_img2.copy()

# Box trigger at top-left corner
poisoned_img1[1:4, 1:4] = 1

# Equal sign trigger at top-left corner
poisoned_img2[2, 1:5] = 1
poisoned_img2[4, 1:5] = 1

# Model Predicted Mask (Flawed Predictions)
predicted_mask1 = poisoned_img1.copy()
predicted_mask2 = poisoned_img2.copy()

# Random incorrect predictions (noise)
noise_indices1 = [(10,10), (15,3), (8,12)]  # 3 random points
noise_indices2 = [(9,9), (14,4), (7,13), (16,6)]  # 4 random points

for idx in noise_indices1:
    predicted_mask1[idx] = 1

for idx in noise_indices2:
    predicted_mask2[idx] = 1

# Plotting
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

ax[0, 0].imshow(clean_img1, cmap='gray')
ax[0, 0].set_title("Clean Image 1")
ax[0, 1].imshow(poisoned_img1, cmap='gray')
ax[0, 1].set_title("Ground Truth Poisoned (Box)")
ax[0, 2].imshow(predicted_mask1, cmap='gray')
ax[0, 2].set_title("Predicted Mask (Box)")

ax[1, 0].imshow(clean_img2, cmap='gray')
ax[1, 0].set_title("Clean Image 2")
ax[1, 1].imshow(poisoned_img2, cmap='gray')
ax[1, 1].set_title("Ground Truth Poisoned (Equal Sign)")
ax[1, 2].imshow(predicted_mask2, cmap='gray')
ax[1, 2].set_title("Predicted Mask (Equal Sign)")

for axes in ax.flatten():
    axes.axis('off')

plt.tight_layout()
plt.show()

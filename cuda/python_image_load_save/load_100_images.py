import torch
from torchvision import datasets, transforms
import numpy as np

# Load MNIST
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Prepare to store flattened images
all_flattened_images = []

# Iterate through the first 100 test images
for i in range(100):
    image, label = test_dataset[i]

    # Convert to numpy and flatten
    img_np = image.numpy().squeeze()  # 28x28
    img_flattened = img_np.flatten()  # 1D array

    # Scale to int16 values by multiplying by 255
    img_scaled = (img_flattened * 255).astype(np.int16)

    all_flattened_images.append(img_scaled)

# Concatenate all 100 flattened and scaled image arrays
combined_images = np.concatenate(all_flattened_images)

# Save the combined NumPy array to a file
np.savetxt("test_images_batch.mem", combined_images, fmt="%d")

print(f"Saved {len(all_flattened_images)} flattened images to test_images_batch.mem")

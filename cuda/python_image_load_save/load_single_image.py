import torch
from torchvision import datasets, transforms
import numpy as np

# Load MNIST
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Get first test image
image, label = test_dataset[0]

print("True label:", label)

# Convert to numpy and scale like FPGA
img_np = image.numpy().squeeze()      # 28x28

img_fixed = (img_np * 255).astype(np.int16)
np.savetxt("test_image.mem", img_fixed.flatten(), fmt="%d")

np.savetxt("test_image.mem", img_fixed.flatten(), fmt="%d")

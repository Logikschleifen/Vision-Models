from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image
import os

# Download FashionMNIST
fashion = datasets.FashionMNIST(
    root='fashion_data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# Directory to save
os.makedirs('fashion_images', exist_ok=True)

# Save first N images
N = 100
for i in range(N):
    img_tensor, label = fashion[i]
    img = to_pil_image(img_tensor)  # convert tensor → PIL Image
    fname = f"{label}_{i}.png"
    img.save(os.path.join('fashion_images', fname))

import os

import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
from torchvision import transforms


def main() -> None:
    model_name = 'WACGAN_2'
    fake_images_dir = f'./results/{model_name}/metrics/fake_images'
    save_path = f'./results/{model_name}/metrics/fake_images_grid.png'

    sample_size = 100
    nrow = 10  # Number of images per row

    # Get all image files from the directory
    image_files = [f for f in os.listdir(
        fake_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = image_files[:sample_size]  # Take first 100 images

    # Transform to convert PIL images to tensors
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load images as tensors
    images = []
    for img_file in image_files:
        img_path = os.path.join(fake_images_dir, img_file)
        img = Image.open(img_path)
        img_tensor = transform(img)
        images.append(img_tensor)

    # Stack images into a batch
    image_batch = torch.stack(images)

    # Create grid using torchvision
    grid = torchvision.utils.make_grid(
        image_batch, nrow=nrow, padding=2, normalize=True)

    # Convert to numpy and display
    grid_np = grid.permute(1, 2, 0).numpy()
    plt.figure(figsize=(15, 15))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()

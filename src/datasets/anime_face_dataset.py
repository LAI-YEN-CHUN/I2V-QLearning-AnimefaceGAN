import os
import shutil

import kagglehub
from torchvision import transforms
from torchvision.datasets import ImageFolder


def download_anime_face_dataset():
    """
    Spencer Churchill/Anime Face Dataset
    Kaggle Directory Structure:
    .
    └── images
        ├── 0_2000.jpg
        ├── 10000_2004.jpg
        ├── 10001_2004.jpg
        ├── 10002_2004.jpg
        └── ...
    """
    handle = 'splcher/animefacedataset'
    download_path = kagglehub.dataset_download(handle)

    return download_path


def move_lung_and_colon_dataset(download_path: str, target_dir: str):
    """
    Move the downloaded dataset to the target directory.
    """
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Move the files from the download path to the target directory
    for filename in os.listdir(download_path):
        source = os.path.join(download_path, filename)
        destination = os.path.join(target_dir, filename)
        shutil.move(source, destination)

    # Remove the original download path
    shutil.rmtree(download_path)


def get_anime_face_dataset(data_dir: str, image_size: int):
    # Define the directory to save the dataset
    raw_data_dir = os.path.join(data_dir, 'anime_face_dataset', 'raw')
    os.makedirs(data_dir, exist_ok=True)

    # Check if the dataset is already downloaded
    if not os.path.exists(raw_data_dir) or len(os.listdir(raw_data_dir)) == 0:
        # Download the dataset
        download_path = download_anime_face_dataset()
        print(f'Downloaded dataset to {download_path}')

        # Move the dataset to the appropriate directories
        move_lung_and_colon_dataset(download_path, raw_data_dir)
        print(f'Moved dataset to {raw_data_dir}')
    else:
        print(f'Dataset already exists at {raw_data_dir}')

    # Define the transformations
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Load the dataset using ImageFolder
    dataset = ImageFolder(root=raw_data_dir, transform=transform)
    print(f'Loaded dataset with {len(dataset)} images')

    return dataset


if __name__ == '__main__':
    # Example usage
    data_dir = './data'
    image_size = 128
    dataset = get_anime_face_dataset(data_dir, image_size)
    print(f'Dataset size: {len(dataset)}')
    print(f'Image size: {dataset[0][0].shape}')

    # Visualize a few images from the dataset
    import matplotlib.pyplot as plt
    import torchvision

    num_rows = 5
    num_samples = num_rows * num_rows
    sample_images = [(dataset[i][0] + 1) / 2 for i in range(num_samples)]  # Normalize to [0, 1]
    grid_images = torchvision.utils.make_grid(
        sample_images, nrow=num_rows, padding=2, normalize=True
    )
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_images.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title('Anime Face Dataset Sample Images')
    plt.show()

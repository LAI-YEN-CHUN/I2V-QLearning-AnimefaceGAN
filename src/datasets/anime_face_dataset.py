import csv
import os
import shutil

import cv2
import kagglehub
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

import illustration2vec.i2v as i2v
from configs import BaseConfig
from utils.utils import is_allowed_file


def download_anime_face_dataset() -> str:
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


def move_anime_face_dataset(download_path: str, target_dir: str) -> None:
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


class AnimeFaceDataset(Dataset):
    def __init__(self, root_dir: str, filenames: list, tags: list, transform=None):
        self.root_dir = root_dir
        self.filenames = filenames
        self.tags = tags
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.filenames[idx])
        # Load image using OpenCV
        image = cv2.imdecode(np.fromfile(
            image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        # Convert from BGR to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert to PyTorch tensor and normalize to [0, 1]
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.tags[idx], dtype=torch.float32)
        return image, label


def data_labeling(cfg: BaseConfig) -> None:
    if not os.path.exists(cfg.PROCESSED_IMAGES_DIR):
        os.makedirs(cfg.PROCESSED_IMAGES_DIR, exist_ok=True)

    if os.listdir(cfg.PROCESSED_IMAGES_DIR):
        print(
            f'Processed image directory {cfg.PROCESSED_IMAGES_DIR} already exists and is not empty.')
        return

    if not os.path.exists(cfg.RAW_IMAGES_DIR) or not os.listdir(cfg.RAW_IMAGES_DIR):
        print(f'Raw image directory {cfg.RAW_IMAGES_DIR} does not exist.')
        return

    # Load i2v model
    print('Loading Illustration2Vec model...')
    illust2vec = i2v.make_i2v_with_chainer(
        cfg.I2V_CAFFEMODEL_PATH, cfg.I2V_TAG_LIST_PATH)
    print('Model loaded.')

    # Remove existing labels CSV if it exists
    if os.path.exists(cfg.LABELS_CSV):
        os.remove(cfg.LABELS_CSV)  # Remove existing file to start fresh

    # Create the labels CSV file and write the header
    fieldnames = ['filename'] + cfg.TARGET_TAGS
    with open(cfg.LABELS_CSV, 'a', newline='', encoding='utf-8') as csvfile:
        # Create a CSV writer object
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process images in batches
        raw_img_files = os.listdir(cfg.RAW_IMAGES_DIR)
        img_files = [f for f in raw_img_files if is_allowed_file(
            f, cfg.ALLOWED_EXTENSIONS)]
        batch_size = cfg.BATCH_SIZE
        batch_count = (len(img_files) + batch_size - 1) // batch_size
        for i in tqdm(range(batch_count), desc='Processing batches'):
            # Calculate batch indices
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(img_files))
            batch_files = img_files[start_idx:end_idx]
            # Load images
            batch_raw_paths = [os.path.join(
                cfg.RAW_IMAGES_DIR, f) for f in batch_files]
            batch_images = [Image.open(p).convert('RGB')
                            for p in batch_raw_paths]
            batch_images = [img.resize(
                (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)) for img in batch_images]
            # Extract labels
            tag_probs_list = illust2vec.estimate_specific_tags(
                batch_images, cfg.TARGET_TAGS)
            # Save labels to CSV
            for img_file, img, tag_probs in zip(batch_files, batch_images, tag_probs_list):
                # Save image
                img.save(os.path.join(cfg.PROCESSED_IMAGES_DIR, img_file))
                # Prepare the row for CSV
                row = {'filename': img_file}
                row.update(tag_probs)
                # Write the row to the CSV file
                writer.writerow(row)


def prepare_anime_face_data(cfg: BaseConfig, num_images: None | int = None) -> tuple:
    # Check if the raw data directory exists and is not empty
    if not os.path.exists(cfg.RAW_IMAGES_DIR) or not os.listdir(cfg.RAW_IMAGES_DIR):
        os.makedirs(cfg.RAW_IMAGES_DIR, exist_ok=True)
        print(
            f'Raw data directory {cfg.RAW_DATA_DIR} does not exist or is empty. Downloading dataset...')
        anime_face_download_path = download_anime_face_dataset()
        print(f"Dataset downloaded to {anime_face_download_path}.")
        move_anime_face_dataset(
            anime_face_download_path, cfg.RAW_DATA_DIR)
        print(f"Download completed. Raw data moved to {cfg.RAW_DATA_DIR}.")

    # Check if the processed data directory exists and is not empty
    if not os.path.exists(cfg.PROCESSED_IMAGES_DIR) or not os.listdir(cfg.PROCESSED_IMAGES_DIR):
        os.makedirs(cfg.PROCESSED_IMAGES_DIR, exist_ok=True)
        print(
            f'Processed data directory {cfg.PROCESSED_DATA_DIR} does not exist or is empty. Preprocessing dataset...')
        data_labeling(cfg)
        print(
            f"Preprocessing completed. Processed data saved to {cfg.PROCESSED_DATA_DIR}.")

    # Load the labels CSV file
    labels_df = pd.read_csv(cfg.LABELS_CSV)

    # Extract filenames and tags from DataFrame
    # DataFrame structure: [filename, tag1, tag2, ...]
    filenames = labels_df['filename'].to_numpy()
    tags = labels_df.drop(columns=['filename']).to_numpy()

    if num_images is not None:
        if num_images > len(filenames):
            raise ValueError(
                f"Requested number of images ({num_images}) exceeds available images ({len(filenames)}).")
        filenames = filenames[:num_images]
        tags = tags[:num_images]

    return filenames, tags


import glob
import os
import time

import cv2
import torch
from pytorch_fid import fid_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import datasets
import models
from configs import BaseConfig
from utils.utils import initialize_directories, set_random_seed
from utils.visualizer import plot_gan_losses


def train_model(cfg: BaseConfig, model: models.WACGAN, dataloader: DataLoader) -> None:
    # Training the model
    start_time = time.time()
    avg_loss_G_list, avg_loss_D_list = model.fit(dataloader)
    training_time = time.time() - start_time
    print(f'Training time: {training_time:.2f} seconds')

    # Save the losses to the log files
    with open(os.path.join(cfg.LOGS_DIR, 'loss_G.csv'), 'w') as f:
        f.write(','.join(map(str, avg_loss_G_list)))
    with open(os.path.join(cfg.LOGS_DIR, 'loss_D.csv'), 'w') as f:
        f.write(','.join(map(str, avg_loss_D_list)))

    # Visualize the losses
    loss_curve_save_path = os.path.join(
        cfg.FIGURES_DIR, 'loss_curve.png')
    plot_gan_losses(avg_loss_G_list, avg_loss_D_list, loss_curve_save_path)


def load_generator(cfg: BaseConfig) -> torch.nn.Module:
    # Load the latest model checkpoint
    checkpoints = glob.glob(os.path.join(
        cfg.GENERATOR_CHECKPOINTS_DIR, "*.pth"))
    print(
        f"Found {len(checkpoints)} checkpoints in {cfg.GENERATOR_CHECKPOINTS_DIR}")
    sorted_checkpoints = sorted(
        checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    if not sorted_checkpoints:
        raise FileNotFoundError("No model checkpoints found.")

    latest_model = sorted_checkpoints[-1]
    print(f"Loading the latest model: {latest_model}")

    model = models.WACGAN_Generator(
        cfg.NOISE_DIM, cfg.CLASS_DIM).to(cfg.DEVICE)
    model.load_state_dict(torch.load(latest_model))
    return model


def compute_fid(real_dataset_dir: str, fake_dataset_dir: str, batch_size: int, device: str):
    fid_value = fid_score.calculate_fid_given_paths(
        [real_dataset_dir, fake_dataset_dir],
        batch_size,
        device,
        dims=2048
    )
    return fid_value


def generate_fake_images(cfg: BaseConfig, generator: torch.nn.Module, fake_y: torch.Tensor) -> None:
    with torch.no_grad():
        generator.eval()
        z = torch.randn(fake_y.shape[0], cfg.NOISE_DIM, device=cfg.DEVICE)
        fake_images = generator(z, fake_y)

    # Save the generated images
    os.makedirs(cfg.FID_FAKE_IMAGES_DIR, exist_ok=True)
    for i, img in enumerate(fake_images):
        img_path = os.path.join(
            cfg.FID_FAKE_IMAGES_DIR, f"fake_image_{i}.png")
        img = (img + 1) / 2  # Normalize to [0, 1]
        img = img.permute(1, 2, 0).cpu().numpy() * 255
        img = img.astype('uint8')
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def copy_real_images(data_loader: DataLoader, target_dir: str) -> None:
    if os.path.exists(target_dir) and os.listdir(target_dir):
        print(
            f"Directory {target_dir} already exists and is not empty. Skipping copy.")
        return

    print(f"Copying images to {target_dir}...")
    os.makedirs(target_dir, exist_ok=True)
    index = 0
    for images, _ in tqdm(data_loader):
        for image in images:
            dst_path = os.path.join(
                target_dir, f"real_image_{index}.png")
            index += 1

            image = (image + 1) / 2  # Normalize to [0, 1]
            image = image.permute(1, 2, 0).cpu().numpy() * 255
            image = image.astype('uint8')
            cv2.imwrite(dst_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def main() -> None:
    cfg = BaseConfig()
    set_random_seed(cfg.RANDOM_SEED)
    initialize_directories(cfg)

    print(f"Using device: {cfg.DEVICE}")

    filenames, tags = datasets.prepare_anime_face_data(cfg)
    print(f"Total images: {len(filenames)}")

    x_train, x_test = filenames[:-cfg.NUM_TEST_IMAGES], \
        filenames[-cfg.NUM_TEST_IMAGES:]
    y_train, y_test = tags[:-cfg.NUM_TEST_IMAGES], tags[-cfg.NUM_TEST_IMAGES:]

    print(f"Training images: {len(x_train)}, Testing images: {len(x_test)}")
    print(f"Training tags: {len(y_train)}, Testing tags: {len(y_test)}")

    transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.AnimeFaceDataset(
        root_dir=cfg.PROCESSED_IMAGES_DIR,
        filenames=x_train,
        tags=y_train,
        transform=transform
    )

    test_dataset = datasets.AnimeFaceDataset(
        root_dir=cfg.PROCESSED_IMAGES_DIR,
        filenames=x_test,
        tags=y_test,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, cfg.BATCH_SIZE,
                              shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, cfg.BATCH_SIZE,
                             shuffle=False, drop_last=False)

    model = models.WACGAN(cfg)

    # Train the model
    print("Starting training...")
    train_model(cfg, model, train_loader)
    print("Training completed.")

    # Load the generator model
    print("Loading the generator model...")
    generator = load_generator(cfg)
    print("Generator model loaded.")

    # Collect all tags from the test loader
    fake_y = torch.cat([tags for _, tags in test_loader], dim=0).to(cfg.DEVICE)
    print(f"Collected {fake_y.shape[0]} tags for fake image generation.")

    # Copy the real images to the FID directory
    print("Copying real images...")
    copy_real_images(test_loader, cfg.FID_REAL_IMAGES_DIR)
    print("Real images copied.")

    # Generate fake images
    print("Generating fake images...")
    generate_fake_images(cfg, generator, fake_y)
    print("Fake images generated.")

    # Evaluate the model
    print("Computing FID score...")
    fid_value = compute_fid(
        cfg.FID_REAL_IMAGES_DIR, cfg.FID_FAKE_IMAGES_DIR, cfg.BATCH_SIZE, cfg.DEVICE)
    print(f"FID score: {fid_value}")


if __name__ == '__main__':
    main()

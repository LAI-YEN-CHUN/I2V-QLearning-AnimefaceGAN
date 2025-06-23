
import glob
import os
import shutil
import time

import cv2
import torch
from pytorch_fid import fid_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import datasets
import models
from configs import BaseConfig
from utils.utils import set_random_seed
from utils.visualizer import plot_gan_losses


def train_model(cfg: BaseConfig, model: models.WACGAN, dataloader: DataLoader) -> None:
    # Training the model
    start_time = time.time()
    avg_loss_G_list, avg_loss_D_list = model.fit(dataloader)
    training_time = time.time() - start_time
    print(f'Training time: {training_time:.2f} seconds')

    # Save the losses to the log files
    if not os.path.exists(cfg.LOGS_DIR):
        os.makedirs(cfg.LOGS_DIR, exist_ok=True)
    with open(os.path.join(cfg.LOGS_DIR, 'loss_G.csv'), 'w') as f:
        f.write(','.join(map(str, avg_loss_G_list)))
    with open(os.path.join(cfg.LOGS_DIR, 'loss_D.csv'), 'w') as f:
        f.write(','.join(map(str, avg_loss_D_list)))

    # Visualize the losses
    if not os.path.exists(cfg.FIGURES_DIR):
        os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    loss_curve_save_path = os.path.join(
        cfg.FIGURES_DIR, 'loss_curve.png')
    plot_gan_losses(avg_loss_G_list, avg_loss_D_list, loss_curve_save_path)


def load_generator(cfg: BaseConfig) -> torch.nn.Module:
    # Load the latest model checkpoint
    checkpoints = glob.glob(os.path.join(
        cfg.GENERATOR_CHECKPOINTS_DIR, "*.pth"))

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
    try:
        fid_value = fid_score.calculate_fid_given_paths(
            [real_dataset_dir, fake_dataset_dir],
            batch_size,
            device,
            dims=2048
        )
    except Exception as e:
        print(f"Error calculating FID: {e}")
        fid_value = float('inf')
    return fid_value


def generate_fake_images(cfg: BaseConfig, generator: torch.nn.Module, fake_y: torch.Tensor) -> None:
    with torch.no_grad():
        generator.eval()
        z = torch.randn(fake_y.shape[0], cfg.NOISE_DIM, device=cfg.DEVICE)
        fake_images = generator(z, fake_y)

    # Save the generated images
    print(f"Saving fake images to {cfg.FID_FAKE_IMAGES_DIR}...")
    os.makedirs(cfg.FID_FAKE_IMAGES_DIR, exist_ok=True)
    for i, img in tqdm(enumerate(fake_images), total=fake_images.shape[0]):
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


def evaluate_model(cfg: BaseConfig, dataloader: DataLoader) -> None:
    # Load the generator model
    generator = load_generator(cfg)

    # Collect all tags from the test loader
    fake_y = torch.cat([tags for _, tags in dataloader], dim=0).to(cfg.DEVICE)

    # Copy the real images to the FID directory
    copy_real_images(dataloader, cfg.FID_REAL_IMAGES_DIR)

    # Generate fake images
    generate_fake_images(cfg, generator, fake_y)

    # Evaluate the model
    fid_value = compute_fid(
        cfg.FID_REAL_IMAGES_DIR, cfg.FID_FAKE_IMAGES_DIR, cfg.BATCH_SIZE, cfg.DEVICE)

    # Save the FID score to a file
    if not os.path.exists(cfg.METRICS_DIR):
        os.makedirs(cfg.METRICS_DIR, exist_ok=True)
    with open(os.path.join(cfg.METRICS_DIR, 'fid_score.txt'), 'w') as f:
        f.write(f"FID score: {fid_value:.4f}\n")

    return fid_value


def main() -> None:
    cfg = BaseConfig(model_name=None)
    set_random_seed(cfg.RANDOM_SEED)

    print(f"Using device: {cfg.DEVICE}")

    filenames, tags = datasets.prepare_anime_face_data(cfg)
    print(f"Total images: {len(filenames)}, shape: {filenames.shape}")
    print(f"Total tags: {len(tags)}, shape: {tags.shape}")

    x_train, x_test, y_train, y_test = train_test_split(
        filenames, tags, test_size=cfg.NUM_TEST_IMAGES, random_state=cfg.RANDOM_SEED
    )

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

    # Remove existing results directory and create a new one
    if os.path.exists(cfg.RESULTS_DIR):
        print(f"Removing existing results directory: {cfg.RESULTS_DIR}...")
        shutil.rmtree(cfg.RESULTS_DIR)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    print("Starting Q-Learning...")
    action2episode = dict()
    agent = models.QAgent(cfg)

    for episode in range(cfg.Q_EPISODES):
        print(f"\n----- Episode {episode + 1} -----")
        model_name = f'WACGAN_{episode + 1}'
        cfg = BaseConfig(model_name=model_name)

        state = agent.get_state()
        action = agent.select_action(state)

        if action is None:
            print("No available actions. Skipping episode.")
            continue

        # Record the action and its corresponding episode
        action2episode[action] = episode
        print(f"Selected action: {action}")

        # Update the configuration with the selected action for training model
        cfg.LEARNING_RATE, cfg.MOMENTUM, cfg.WEIGHT_DECAY = action

        # Save the hyperparameters for the current episode
        if not os.path.exists(cfg.RESULTS_DIR):
            os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
        with open(os.path.join(cfg.RESULTS_DIR, 'hyperparameters.txt'), 'w') as f:
            f.write(f"Episode: {episode + 1}\n")
            f.write(f"Learning Rate: {cfg.LEARNING_RATE}\n")
            f.write(f"Momentum: {cfg.MOMENTUM}\n")
            f.write(f"Weight Decay: {cfg.WEIGHT_DECAY}\n")

        # Initialize the model with the selected hyperparameters
        model = models.WACGAN(cfg)

        print("Training the model...")
        train_model(cfg, model, train_loader)

        print("Evaluating the model...")
        fid_value = evaluate_model(cfg, test_loader)
        print(f"FID score for episode {episode + 1}: {fid_value:.4f}")

        reward = -fid_value

        # Get the next state (The stateless agent does not change state)
        next_state = agent.get_state()

        # If the reward is finite, update the Q-table
        if reward != float('-inf'):
            agent.update_q(state, action, reward, next_state)

    print("Q-Learning completed.\n")

    # Print the final Q-table
    print("Final Q-table:")
    for state, actions in agent.q_table.items():
        print(f"State: {state}")
        for action, value in actions.items():
            print(f"  Action: {action}, Q-value: {value:.4f}")

    # Find the best action
    best_action = None
    best_value = float('-inf')
    for state, actions in agent.q_table.items():
        for action, value in actions.items():
            if value > best_value:
                best_value = value
                best_action = action

    if best_action is None:
        print("No best action found. Exiting.")
        return

    best_episode = action2episode[best_action]
    best_model_name = f'WACGAN_{best_episode + 1}'
    print(f"Best action: {best_action} from episode {best_episode + 1}")
    print(f"Best model name: {best_model_name}")
    print(
        f"Best hyperparameters: Learning Rate: {best_action[0]}, Momentum: {best_action[1]}, Weight Decay: {best_action[2]}")

    # Load the best model configuration
    cfg = BaseConfig(model_name=best_model_name)


if __name__ == '__main__':
    main()

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import BaseConfig
from models.common import compute_gradient_penalty, weights_init


class WACGAN_Generator(nn.Module):
    """
    Conditional Generator Network with reduced layers
    input shape: (batch_size, noise_dim)
    output shape: (batch_size, 3, 64, 64)
    """

    def __init__(self, noise_dim: int, class_dim: int, feature_map_size: int = 64):
        super(WACGAN_Generator, self).__init__()
        image_channels = 3  # Number of channels in the image

        self.model = nn.Sequential(
            # (batch_size, noise_dim + class_dim, 1, 1) -> (batch_size, feature_map_size*8, 4, 4)
            nn.ConvTranspose2d(in_channels=noise_dim + class_dim, out_channels=feature_map_size*8,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_map_size*8),
            nn.ReLU(inplace=True),
            # (batch_size, feature_map_size*8, 4, 4) -> (batch_size, feature_map_size*4, 8, 8)
            nn.ConvTranspose2d(in_channels=feature_map_size*8, out_channels=feature_map_size*4,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size*4),
            nn.ReLU(inplace=True),
            # (batch_size, feature_map_size*4, 8, 8) -> (batch_size, feature_map_size*2, 16, 16)
            nn.ConvTranspose2d(in_channels=feature_map_size*4, out_channels=feature_map_size*2,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size*2),
            nn.ReLU(inplace=True),
            # (batch_size, feature_map_size*2, 16, 16) -> (batch_size, feature_map_size, 32, 32)
            nn.ConvTranspose2d(in_channels=feature_map_size*2, out_channels=feature_map_size,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(inplace=True),
            # (batch_size, feature_map_size, 32, 32) -> (batch_size, image_channels, 64, 64)
            nn.ConvTranspose2d(in_channels=feature_map_size, out_channels=image_channels,
                               kernel_size=4, stride=2, padding=1, bias=False),
            # Apply Tanh activation
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, z, c):
        # z: (batch_size, noise_dim), c: (batch_size, class_dim)
        # (batch_size, noise_dim + class_dim)
        x = torch.cat([z, c], dim=1)
        # (batch_size, noise_dim + class_dim, 1, 1)
        x = x.view(x.size(0), -1, 1, 1)
        y = self.model(x)  # (batch_size, 3, 64, 64)
        return y


class WACGAN_Discriminator(nn.Module):
    """
    Conditional Discriminator Network with reduced layers
    input shape: (batch_size, 3, 64, 64), (batch_size, 2)
    output shape: (batch_size, 1)
    """

    def __init__(self, class_dim: int, feature_map_size: int = 64):
        super(WACGAN_Discriminator, self).__init__()
        image_channels = 3  # Number of channels in the image

        self.model = nn.Sequential(
            # (batch_size, image_channels + class_dim, 64, 64) -> (batch_size, feature_map_size, 32, 32)
            nn.Conv2d(in_channels=image_channels + class_dim, out_channels=feature_map_size,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (batch_size, feature_map_size, 32, 32) -> (batch_size, feature_map_size*2, 16, 16)
            nn.Conv2d(in_channels=feature_map_size, out_channels=feature_map_size*2,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size*2),
            nn.LeakyReLU(0.2, inplace=True),
            # (batch_size, feature_map_size*2, 16, 16) -> (batch_size, feature_map_size*4, 8, 8)
            nn.Conv2d(in_channels=feature_map_size*2, out_channels=feature_map_size*4,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size*4),
            nn.LeakyReLU(0.2, inplace=True),
            # (batch_size, feature_map_size*4, 8, 8) -> (batch_size, feature_map_size*8, 4, 4)
            nn.Conv2d(in_channels=feature_map_size*4, out_channels=feature_map_size*8,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size*8),
            nn.LeakyReLU(0.2, inplace=True),
            # (batch_size, feature_map_size*8, 4, 4) -> (batch_size, 1, 1, 1)
            nn.Conv2d(in_channels=feature_map_size*8, out_channels=1,
                      kernel_size=4, stride=1, padding=0, bias=False),
            # WGAN does not use Sigmoid activation
        )

        self.apply(weights_init)

    def forward(self, x, c):
        # x: (batch_size, 3, 64, 64), c: (batch_size, class_dim)

        # Expand `c` to match the spatial dimensions of `x`
        # (batch_size, class_dim) -> (batch_size, class_dim, 1, 1)
        c = c.view(c.size(0), c.size(1), 1, 1)  #
        # (batch_size, class_dim, 1, 1) -> (batch_size, class_dim, 64, 64)
        c = c.expand(-1, -1, x.size(2), x.size(3))

        # Concatenate `c` with `x` along the channel dimension
        # (batch_size, image_channels + class_dim, 64, 64)
        x = torch.cat([x, c], dim=1)

        y = self.model(x)  # (batch_size, 1, 1, 1)
        return y.view(-1, 1)  # Reshape to (batch_size, 1)


class WACGAN():
    def __init__(self, cfg: BaseConfig):
        self.generator = WACGAN_Generator(cfg.NOISE_DIM, cfg.CLASS_DIM)
        self.discriminator = WACGAN_Discriminator(cfg.CLASS_DIM)
        self.cfg = cfg

        self.optimizer_G = torch.optim.RMSprop(
            self.generator.parameters(),
            lr=cfg.LEARNING_RATE,
            momentum=cfg.MOMENTUM,
            weight_decay=cfg.WEIGHT_DECAY
        )
        self.optimizer_D = torch.optim.RMSprop(
            self.discriminator.parameters(),
            lr=cfg.LEARNING_RATE,
            momentum=cfg.MOMENTUM,
            weight_decay=cfg.WEIGHT_DECAY
        )

    def fit(self, data_loader: DataLoader) -> Tuple[List[float], List[float]]:
        # Set the device for the training
        device = self.cfg.DEVICE
        self.generator.to(device)
        self.discriminator.to(device)

        # Lists to store the losses
        avg_loss_G_list = []
        avg_loss_D_list = []

        for epoch in range(self.cfg.EPOCHES):
            total_loss_G = 0  # Total loss for the generator
            total_loss_D = 0  # Total loss for the discriminator
            data_count = 0  # Total number of data points processed
            progress_bar = tqdm(data_loader)
            for batch_idx, (real_x, real_y) in enumerate(progress_bar):
                # Set the models in training mode
                self.generator.train()
                self.discriminator.train()
                # Move the data to the device
                real_x, real_y = real_x.to(device), real_y.to(device)
                # Get the batch size
                batch_size = real_x.size(0)

                # ============================================
                #  Train D
                # ============================================
                # Generate noise samples
                z = torch.randn(batch_size, self.cfg.NOISE_DIM).to(device)
                # Get the generated samples
                fake_x = self.generator(z, real_y).detach()
                # Compute the output of the discriminator
                real_output = self.discriminator(real_x, real_y)
                fake_output = self.discriminator(fake_x, real_y)
                # Compute the gradient penalty
                gradient_penalty = compute_gradient_penalty(
                    self.discriminator, real_x.data, fake_x.data, device, labels=real_y)
                # Compute the loss for the discriminator
                loss_D = -torch.mean(real_output) + torch.mean(fake_output) + \
                    self.cfg.LAMBDA_GP * gradient_penalty
                # Update the discriminator
                self.discriminator.zero_grad()
                loss_D.backward()
                self.optimizer_D.step()

                # ============================================
                #  Train G
                # ============================================
                if batch_idx % self.cfg.CRITICS == 0:
                    # Generate noise samples
                    z = torch.randn(batch_size, self.cfg.NOISE_DIM).to(device)
                    # Get the generated samples
                    fake_x = self.generator(z, real_y)
                    # Compute the output of the discriminator
                    fake_output = self.discriminator(fake_x, real_y)
                    # Compute the loss for the generator
                    loss_G = -torch.mean(fake_output)
                    # Update the generator
                    self.generator.zero_grad()
                    loss_G.backward()
                    self.optimizer_G.step()

                # Update the losses and data count
                total_loss_D += loss_D.item() * batch_size
                total_loss_G += loss_G.item() * batch_size
                data_count += batch_size

                # Update the progress bar description
                progress_bar.set_description(
                    f'Epoch [{epoch+1}/{self.cfg.EPOCHES}]'
                    f' | Loss D: {total_loss_D / data_count:.4f}'
                    f' | Loss G: {total_loss_G / data_count:.4f}'
                )
            # Set the models in evaluation mode
            self.generator.eval()
            self.discriminator.eval()

            # Compute the average loss for the generator and discriminator
            avg_loss_G = total_loss_G / data_count
            avg_loss_D = total_loss_D / data_count
            avg_loss_G_list.append(avg_loss_G)
            avg_loss_D_list.append(avg_loss_D)

            # Ensure the directories for saving checkpoints and samples exist
            if not os.path.exists(self.cfg.GENERATOR_CHECKPOINTS_DIR):
                os.makedirs(self.cfg.GENERATOR_CHECKPOINTS_DIR,
                            exist_ok=True)
            if not os.path.exists(self.cfg.DISCRIMINATOR_CHECKPOINTS_DIR):
                os.makedirs(
                    self.cfg.DISCRIMINATOR_CHECKPOINTS_DIR, exist_ok=True)
            if not os.path.exists(self.cfg.SAMPLES_DIR):
                os.makedirs(self.cfg.SAMPLES_DIR, exist_ok=True)

            # Save the model checkpoints
            if (epoch + 1) % self.cfg.MODEL_SAVE_INTERVAL == 0 or epoch == self.cfg.EPOCHES - 1:
                generator_save_path = os.path.join(
                    self.cfg.GENERATOR_CHECKPOINTS_DIR, f'generator_{epoch + 1}.pth')
                discriminator_save_path = os.path.join(
                    self.cfg.DISCRIMINATOR_CHECKPOINTS_DIR, f'discriminator_{epoch + 1}.pth')
                torch.save(self.generator.state_dict(),
                           generator_save_path)
                torch.save(self.discriminator.state_dict(),
                           discriminator_save_path)

            # Generate samples
            if (epoch + 1) % self.cfg.SAMPLE_INTERVAL == 0 or epoch == self.cfg.EPOCHES - 1:
                samples_save_path = os.path.join(
                    self.cfg.SAMPLES_DIR, f'samples_{epoch + 1}.png')
                self.__plot_samples(samples_save_path)

        return avg_loss_G_list, avg_loss_D_list

    def __plot_samples(self, save_path: str) -> None:
        with torch.no_grad():
            num_permutations = self.cfg.NUM_SEX * \
                self.cfg.NUM_HAIR_COLORS * self.cfg.NUM_EYE_COLORS
            sex_y = torch.eye(self.cfg.NUM_SEX, device=self.cfg.DEVICE)
            hair_y = torch.eye(self.cfg.NUM_HAIR_COLORS,
                               device=self.cfg.DEVICE)
            eye_y = torch.eye(self.cfg.NUM_EYE_COLORS, device=self.cfg.DEVICE)
            fake_y = [
                torch.cat((sex_y[s], hair_y[h], eye_y[e]), dim=0)
                for s in range(self.cfg.NUM_SEX)
                for h in range(self.cfg.NUM_HAIR_COLORS)
                for e in range(self.cfg.NUM_EYE_COLORS)
            ]
            fake_y = torch.stack(fake_y).to(self.cfg.DEVICE)

            z = torch.randn(num_permutations, self.cfg.NOISE_DIM).to(
                self.cfg.DEVICE)
            fake_samples = self.generator(z, fake_y)

        # Save the generated samples
        fig, axs = plt.subplots(20, 10, figsize=(20, 40))
        for i in range(self.cfg.NUM_SEX):
            for j in range(self.cfg.NUM_HAIR_COLORS):
                for k in range(self.cfg.NUM_EYE_COLORS):
                    idx = i * self.cfg.NUM_HAIR_COLORS * \
                        self.cfg.NUM_EYE_COLORS + j * self.cfg.NUM_EYE_COLORS + k
                    ax = axs[i * self.cfg.NUM_HAIR_COLORS + j, k]
                    img = fake_samples[idx].detach().cpu().numpy()
                    img = (img + 1) / 2  # Scale the pixel values to [0, 1]
                    if img.shape[0] == 1:  # Grayscale image
                        ax.imshow(img[0], cmap='gray')
                    else:  # Color image
                        ax.imshow(img.transpose(1, 2, 0))
                    ax.axis('off')

                    # Set title with the attributes
                    sex_type = self.cfg.TARGET_TAGS[i]
                    # +2 to skip the first two sex tags
                    hair_color = self.cfg.TARGET_TAGS[j + 2]
                    # +12 to skip sex and hair tags
                    eye_color = self.cfg.TARGET_TAGS[k + 12]

                    # For first column, add hair color label on y-axis
                    if k == 0:
                        ax.text(-0.1, 0.5, hair_color,
                                transform=ax.transAxes, rotation=0,
                                va='center', ha='right', fontsize=10)

                    # For first row of each gender section, add eye color label on x-axis
                    if i == 0 and j == 0:
                        ax.text(0.5, 1.1, eye_color,
                                transform=ax.transAxes, rotation=0,
                                va='bottom', ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

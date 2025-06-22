from typing import List

import matplotlib.pyplot as plt


def plot_gan_losses(loss_G_list: List[float], loss_D_list: List[float], save_path: str | None = None) -> None:
    """Visualize the losses of the GAN model"""
    plt.figure(figsize=(10, 5))
    plt.plot(loss_G_list, label='Generator Loss', color='red')
    plt.plot(loss_D_list, label='Discriminator Loss', color='blue')
    plt.legend()
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

import numpy as np
import torch


def compute_gradient_penalty(
    discriminator: torch.nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: torch.device,
    labels: torch.Tensor | None = None
) -> torch.Tensor:
    # Get the batch size of the samples
    batch_size = real_samples.size(0)
    # Random weight term for interpolation between real and fake samples
    shape_ones = np.ones(len(real_samples.size()) - 1, dtype=int)
    epsilon = torch.rand((batch_size, *shape_ones)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolates.requires_grad_(True)
    # Get the discriminator output for the interpolated samples
    if labels is not None:
        d_interpolated = discriminator(interpolates, labels)
    else:
        d_interpolated = discriminator(interpolates)
    # Get gradient w.r.t. interpolates
    # Gradient penalty requires the gradient w.r.t the discriminator to be 1
    fake = torch.ones(batch_size, 1).to(device)
    fake.requires_grad = False
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )
    # Reshape the gradients to take the norm
    gradients = gradients[0].view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

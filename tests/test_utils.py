import einops
import torch

from torch_subtract_membranes_2d.path_models.path_2d import Path2D
from torch_subtract_membranes_2d.utils.path_utils import sample_image_along_path
import torch_grid_utils

def test_sample_image_along_path():
    circle_image = torch_grid_utils.circle(
        radius=10,
        image_shape=(32, 32),
        center=(16, 16)
    )
    theta = torch.linspace(0, 2*torch.pi, steps=100)
    x = 10 * torch.cos(theta) + 16
    y = 10 * torch.sin(theta) + 16
    control_points = einops.rearrange([y, x], "yx b -> b yx")
    path = Path2D(control_points=control_points, is_closed=True, yx_coords=True)
    perpendicular_steps = torch.arange(start=-5, end=6, step=1)  # [-5, ..., 5]
    samples, mask = sample_image_along_path(image=circle_image, path=path, n_samples=100, distances=perpendicular_steps)

    assert torch.allclose(samples[:, :4], torch.ones_like(samples[:, :4]), atol=1e-5)
    assert torch.allclose(samples[:, 7:], torch.zeros_like(samples[:, 7:]), atol=1e-5)
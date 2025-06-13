import einops
import torch
from torch_image_interpolation import sample_image_1d

from torch_subtract_membranes_2d.membrane_model import Membrane2D
from torch_subtract_membranes_2d.utils.path_utils import find_pixels_around_path


def render_membrane_image(
    membranes: list[Membrane2D],
    image_shape: tuple[int, int]
) -> torch.Tensor:
    # create image for output
    membrane_image = torch.zeros(image_shape, dtype=torch.float32)

    # render membranes one by one
    for idx, membrane in enumerate(membranes):
        membrane_image += _render_single_membrane_image(
            membrane=membrane, image_shape=image_shape
        )
    return membrane_image


def _render_single_membrane_image(
    membrane: Membrane2D,
    image_shape: tuple[int, int],
    device: torch.device | None = None,
) -> torch.Tensor:
    # create image for output
    image = torch.zeros(image_shape, dtype=torch.float32, device=device)

    # find membrane pixel positions
    maximum_distance = (len(membrane.profile_1d) // 2) + 2
    idx_h, idx_w = find_pixels_around_path(
        path=membrane.path,
        image_shape=image_shape,
        maximum_distance=maximum_distance
    )
    membrane_pixel_positions = einops.rearrange([idx_h, idx_w], "yx b -> b yx")

    # get signed distance from membrane at each pixel position
    closest_u = membrane.path.get_closest_u(
        query_points=membrane_pixel_positions
    )
    signed_distances = membrane.path.get_signed_distance(
        query_points=membrane_pixel_positions,
        closest_u=closest_u,
    )

    # find corresponding positions to sample from in 1d profiles
    center = len(membrane.profile_1d) // 2
    sample_positions_1d = center + signed_distances

    # sample values from average and weights for each pixel
    values = sample_image_1d(
        image=torch.as_tensor(membrane.profile_1d, dtype=torch.float32, device=device),
        coordinates=sample_positions_1d,
        interpolation="cubic",
    )
    weights = sample_image_1d(
        image=torch.as_tensor(membrane.weights_1d, dtype=torch.float32, device=device),
        coordinates=sample_positions_1d,
        interpolation="cubic",
    )

    # interpolate a signal scale value for each pixel
    signal_scale = membrane.signal_scale_spline.interpolate(u=closest_u)

    # place values into image
    image[idx_h, idx_w] += signal_scale * values * weights

    return image

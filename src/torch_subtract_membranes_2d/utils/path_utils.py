import einops
import numpy as np
import torch
import scipy.ndimage as ndi
from torch_image_interpolation import sample_image_2d, insert_into_image_1d

from torch_subtract_membranes_2d.path_models.path_2d import Path2D


def rasterize_path(
        path: Path2D,
        image_shape: tuple[int, int],
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
):
    # oversample path relative to pixel spacing to ensure no gaps
    n_samples = int(100 * path.estimated_length())
    u = torch.linspace(0, 1, steps=n_samples, device=device)
    points = path.interpolate(u=u)  # (b, 2) yx coords

    # include only points that are inside the image
    upper_bound = torch.tensor(image_shape, device=device) - 1
    idx_inside = torch.logical_and(points > 0, points < upper_bound)
    idx_inside = torch.all(idx_inside, dim=-1)
    points = points[idx_inside]

    # find pixel indices for each point
    nearest = torch.round(points).int()
    idx_h, idx_w = einops.rearrange(nearest, "b yx -> yx b")

    # place 1 at each pixel
    rasterized_path = torch.zeros(image_shape, dtype=dtype, device=device)
    ones = torch.ones(len(points), dtype=dtype, device=device)
    rasterized_path.index_put_(indices=(idx_h, idx_w), values=ones, accumulate=False)
    return rasterized_path


def find_pixels_around_path(
        path: Path2D,
        image_shape: tuple[int, int],
        maximum_distance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # rasterize path
    device = path.control_points.device
    rasterized_path = rasterize_path(path=path, image_shape=image_shape, device=device)
    rasterized_path = rasterized_path.detach().cpu().numpy().astype(np.bool)

    # find distance from every pixel to the path
    distance_transform = ndi.distance_transform_edt(~rasterized_path)  # (h, w)

    # find pixel indices for all pixels closer than distance threshold
    idx_h, idx_w = np.where(distance_transform <= maximum_distance)
    idx_h, idx_w = torch.as_tensor(idx_h), torch.as_tensor(idx_w)
    return idx_h, idx_w


def calculate_1d_average_along_path(
        path: Path2D,
        image: torch.Tensor,
        maximum_distance: int
) -> torch.Tensor:
    # setup array for reconstruction
    reconstruction_size = 2 * maximum_distance
    reconstruction_center = maximum_distance
    reconstruction_1d = torch.zeros(size=(reconstruction_size,))

    # find membrane pixels
    idx_h, idx_w = find_pixels_around_path(
        path=path,
        image_shape=image.shape,
        maximum_distance=maximum_distance
    )
    membrane_pixel_yx = einops.rearrange([idx_h, idx_w], "yx b -> b yx")

    # get signed distances to each pixel
    signed_distance = path.get_signed_distance(query_points=membrane_pixel_yx)

    # do reconstruction with linear interpolation
    reconstruction_1d, weights = insert_into_image_1d(
        values=image[idx_h, idx_w],
        coordinates=signed_distance + reconstruction_center,
        image=reconstruction_1d,
        interpolation="linear",
    )
    reconstruction_1d = reconstruction_1d / weights

    return reconstruction_1d


def sample_image_along_path(
        path: Path2D,
        image: torch.Tensor,
        distances: torch.Tensor,
        n_samples: int
) -> torch.Tensor:
    """Sample an image along a path.

    Parameters
    ----------
    path: Path2D
        The path to be sampled.
    image: torch.Tensor
        `(h, w)` array containing image data
    distances: torch.Tensor
        `(w, )` array of signed distances from the path at which to sample
    n_samples: int
        Number of segments perpendicular to the path to generate

    Returns
    -------
    (result, mask): torch.Tensor
        `(n_samples, w)` array containing `w` samples at `n_samples` points along the path.
    """
    # grab device and force signed distance vector onto device
    device = image.device
    distances = distances.to(device)

    # sample points and normals along path
    u = torch.linspace(0, 1, n_samples, device=device)
    points = path.interpolate(u)  # (n_samples, 2)
    normals = path.get_normals(u)  # (n_samples, 2)

    # Generate sampling coordinates along normals for each point
    # shape: (n_samples, n_perpendicular_steps, 2)
    h, w = image.shape[-2:]
    h_out, w_out = n_samples, len(distances)
    points = einops.repeat(points, "h yx -> h w yx", w=w_out)
    distances = einops.repeat(distances, "w -> h w yx", h=h_out, yx=2)
    normals = einops.repeat(normals, "h yx -> h w yx", w=w_out)
    sampling_coords = points + distances * normals

    # take samples
    result = sample_image_2d(image=image, coordinates=sampling_coords)

    # construct mask for in/out of bounds samples
    upper_bound = torch.tensor([h - 1, w - 1], device=device)
    mask = torch.logical_and(sampling_coords >= 0, sampling_coords <= upper_bound)
    mask = torch.all(mask, dim=-1)

    return result, mask

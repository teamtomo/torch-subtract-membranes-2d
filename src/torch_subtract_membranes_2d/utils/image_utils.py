import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch_fourier_filter.bandpass import bandpass_filter


def normalize_2d(image: torch.Tensor):
    """normalize image to mean 0, stddev 1 based on central 50%"""
    h, w = image.shape[-2:]
    hl, hu = int(0.25 * h), int(0.75 * h)
    wl, wu = int(0.25 * w), int(0.75 * w)
    center = image[..., hl:hu, wl:wu]
    image = (image - torch.mean(center)) / torch.std(center)
    return image


def bandpass_filter_image(
    image: torch.Tensor,
    pixel_spacing: float,
    highpass_angstroms: float | None = None,
    lowpass_angstroms: float | None = None,
):
    # pad image
    h, w = image.shape[-2:]
    dh, dw = h // 2, w // 2
    image = einops.rearrange(image, "h w -> 1 h w")
    image = F.pad(image, pad=(dw, dw, dh, dh), mode="replicate")
    image = einops.rearrange(image, "1 h w -> h w")

    # construct filter
    # low frequency cuton
    if highpass_angstroms is None:
        low = 0
    else:
        low = spatial_frequency_to_fftfreq(1 / highpass_angstroms, spacing=pixel_spacing)

    # high frequency cutoff
    if lowpass_angstroms is None:
        high = 0.5
    else:
        high = spatial_frequency_to_fftfreq(1 / lowpass_angstroms, spacing=pixel_spacing)

    # falloff
    falloff = spatial_frequency_to_fftfreq(1 / (2 * highpass_angstroms), spacing=pixel_spacing)

    filter = bandpass_filter(
        low=low,
        high=high,
        falloff=falloff,
        image_shape=image.shape[-2:],
        rfft=True,
        fftshift=False,
        device=image.device,
    )

    # apply filter to dft
    dft = torch.fft.rfftn(image, dim=(-2, -1))
    dft *= filter

    # transform back to real space and return
    filtered_image = torch.fft.irfftn(dft, dim=(-2, -1))
    filtered_image = filtered_image[..., dh:-dh, dw:-dw]
    return filtered_image


def fftfreq_to_spatial_frequency(
    frequencies: torch.Tensor, spacing: float = 1
) -> torch.Tensor:
    """Convert frequencies in cycles per pixel to cycles per unit distance."""
    # cycles/px * px/distance = cycles/distance
    return torch.as_tensor(frequencies, dtype=torch.float32) * (1 / spacing)


def spatial_frequency_to_fftfreq(
    frequencies: torch.Tensor, spacing: float = 1
) -> torch.Tensor:
    """Convert frequencies in cycles per unit distance to cycles per pixel."""
    # cycles/distance * distance/px = cycles/px
    return torch.as_tensor(frequencies, dtype=torch.float32) * spacing


def smooth_tophat_1d(
    length: int,
    center_width: int,
    rolloff_width: int,
    device=None
) -> torch.Tensor:
    """
    Create a smooth top-hat weighting function.

    Args:
        length: Total length of the weighting function
        center_width: Width of the center region with value 1
        rolloff_width: Width of the rolloff region on each side
        device: Device to place the tensor on

    Returns:
        Tensor of shape (length, ) with the smooth top-hat weights
    """
    if center_width + 2 * rolloff_width > length:
        raise ValueError("center_width + 2*rolloff_width must be <= length")

    weights = torch.zeros(length, device=device)

    # Calculate center region
    center_start = (length - center_width) // 2
    center_end = center_start + center_width

    # Set center region to 1
    weights[center_start:center_end] = 1.0

    # Create left rolloff (using cosine rolloff)
    if rolloff_width > 0:
        left_start = center_start - rolloff_width
        left_end = center_start
        if left_start >= 0:
            x = torch.linspace(np.pi, 0, rolloff_width, device=device)
            rolloff_values = 0.5 * (1 + torch.cos(x))
            weights[left_start:left_end] = rolloff_values

        # Create right rolloff
        right_start = center_end
        right_end = center_end + rolloff_width
        if right_end <= length:
            x = torch.linspace(0, np.pi, rolloff_width, device=device)
            rolloff_values = 0.5 * (1 + torch.cos(x))
            weights[right_start:right_end] = rolloff_values

    return weights

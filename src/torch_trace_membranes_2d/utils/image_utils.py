import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch_fourier_filter.bandpass import bandpass_filter
from torch_fourier_shift import fourier_shift_image_1d


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
    print(falloff)


    filter = bandpass_filter(
        low=low,
        high=high,
        falloff=falloff,
        image_shape=image.shape[-2:],
        rfft=True,
        fftshift=False
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


def center_of_mass_1d(image: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(len(image), dtype=image.dtype, device=image.device)
    image = image - torch.min(image)
    total_mass = torch.sum(image)
    weighted_positions = torch.sum(positions * image)
    center_of_mass = weighted_positions / total_mass
    return center_of_mass


def center_1d_profile(image: torch.Tensor) -> torch.Tensor:
    # grab signal length
    w = len(image)
    # estimate background
    background = apply_gaussian_filter_1d(
        signal=image,
        kernel_size=w // 3,
        sigma=w / 3,
    )
    image_no_center = image.clone()
    cutoff = int(0.25 * w)
    mean = torch.mean(torch.cat([image[:cutoff], image[-cutoff:]]))
    image_no_center[cutoff:-cutoff] = mean
    dft = torch.fft.rfft(image_no_center, dim=-1)
    idx_high = torch.fft.rfftfreq(w) > 0.025
    dft[idx_high] = 0
    background = torch.fft.irfft(dft, dim=-1, n=w)

    # calculate center of mass after subtracting background
    background_subtracted = image - background
    background_subtracted_positive_membrane = -1 * background_subtracted
    center_of_mass = center_of_mass_1d(background_subtracted_positive_membrane)

    # from matplotlib import pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(image.detach(), label="image")
    # ax.plot(image_no_center.detach(), label="image (membrane erased)")
    # ax.plot(background.detach(), label="background")
    # ax.plot((image - background).detach(), label="image - background")
    # # ax.stem(center_of_mass.detach(), torch.max(image).detach(), label="center of mass")
    # ax.legend(loc="best")
    # plt.show()

    # calculate how much to shift profile by
    center = len(image) / 2
    shift = -1 * (center_of_mass - center)
    shift = torch.as_tensor(shift, device=image.device)

    # center the profile
    centered_image = fourier_shift_image_1d(image=image, shifts=shift)
    return centered_image


def gaussian_kernel_1d(kernel_size, sigma):
    """
    Create a 1D Gaussian kernel.

    Args:
        kernel_size (int): Size of the kernel (should be odd)
        sigma (float): Standard deviation of the Gaussian

    Returns:
        torch.Tensor: 1D Gaussian kernel
    """
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create coordinate grid
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords -= kernel_size // 2

    # Compute Gaussian values
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))

    # Normalize so sum equals 1
    kernel = kernel / kernel.sum()

    return kernel


def apply_gaussian_filter_1d(signal, kernel_size=5, sigma=1.0):
    # (w, ) -> (c, h, w)
    signal = einops.rearrange(signal, "w -> 1 1 w")

    # Create Gaussian kernel with (c, h, w)
    kernel = gaussian_kernel_1d(kernel_size, sigma)
    kernel = einops.rearrange(kernel, "w -> 1 1 w")

    # Apply convolution
    pad = kernel_size // 2
    filtered = F.conv1d(signal, kernel, padding=pad)

    # (c, h, w) -> (w, )
    filtered = einops.rearrange(filtered, "1 1 w -> w")
    return filtered

import torch

from torch_subtract_membranes_2d.constants import REFINEMENT_HIGHPASS_ANGSTROMS
from torch_subtract_membranes_2d.membrane_model import Membrane2D
from torch_subtract_membranes_2d.utils import IS_DEBUG
from torch_subtract_membranes_2d.utils.debug_utils import set_matplotlib_resolution
from torch_subtract_membranes_2d.utils.image_utils import bandpass_filter_image, normalize_2d
from torch_subtract_membranes_2d.render_membranes import render_membrane_image


def subtract_membranes(
    image: torch.Tensor,
    pixel_spacing_angstroms: float,
    membranes: list[Membrane2D],
    subtraction_factor: float = 1.0
) -> torch.Tensor:
    # grab image dimensions
    h, w = image.shape[-2:]

    # normalize and highpass as in `trace_membranes`
    image = normalize_2d(image)
    image = bandpass_filter_image(
        image=image,
        pixel_spacing=pixel_spacing_angstroms,
        highpass_angstroms=REFINEMENT_HIGHPASS_ANGSTROMS,
        lowpass_angstroms=None,
    )

    # render membrane image
    membrane_image = render_membrane_image(
        membranes=membranes,
        image_shape=(h, w),
        device=image.device,
    )

    # subtract membrane image from bandpassed image
    subtracted = image - (subtraction_factor * membrane_image)

    if IS_DEBUG:
        from torch_fourier_rescale import fourier_rescale_2d
        from matplotlib import pyplot as plt

        image_downscaled, _ = fourier_rescale_2d(image, source_spacing=1, target_spacing=12)
        membrane_image_downscaled, _ = fourier_rescale_2d(membrane_image, source_spacing=1, target_spacing=12)
        subtracted_downscaled, _ = fourier_rescale_2d(subtracted, source_spacing=1, target_spacing=12)

        fig, axs = plt.subplots(ncols=3, figsize=(12, 4))
        for ax in axs:
            ax.set_axis_off()
        axs[0].set_title("image")
        axs[0].imshow(image_downscaled.detach().cpu().numpy(), cmap="gray")
        axs[1].set_title("2D reconstruction")
        axs[1].imshow(membrane_image_downscaled.detach().cpu().numpy(), cmap="gray")
        axs[2].set_title(f"image\n({int(100 * subtraction_factor)}% subtracted)")
        axs[2].imshow(subtracted_downscaled.detach().cpu().numpy(), cmap="gray")
        plt.tight_layout()
        plt.show()

    return subtracted

if IS_DEBUG:
    set_matplotlib_resolution()
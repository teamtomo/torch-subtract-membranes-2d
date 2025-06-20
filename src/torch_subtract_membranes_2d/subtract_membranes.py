import os
from datetime import datetime
from pathlib import Path

import torch

from torch_subtract_membranes_2d.constants import REFINEMENT_HIGHPASS_ANGSTROMS
from torch_subtract_membranes_2d.membrane_model import Membrane2D
from torch_subtract_membranes_2d.utils import IS_DEBUG
from torch_subtract_membranes_2d.utils.plotting_utils import plot_subtraction_results
from torch_subtract_membranes_2d.utils.image_utils import bandpass_filter_image, normalize_2d
from torch_subtract_membranes_2d.utils.datetime_utils import humanize_timedelta
from torch_subtract_membranes_2d.render_membranes import render_membrane_image


def subtract_membranes(
    image: torch.Tensor,
    pixel_spacing_angstroms: float,
    membranes: list[Membrane2D],
    subtraction_factor: float = 1.0,
    output_image_directory: os.PathLike | None = None,
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
    start = datetime.now()
    membrane_image = render_membrane_image(
        membranes=membranes,
        image_shape=(h, w),
        device=image.device,
    )

    # subtract membrane image from bandpassed image
    subtracted = image - (subtraction_factor * membrane_image)
    end = datetime.now()
    print(f"time taken for membrane subtraction: {humanize_timedelta(end - start)}")

    # plot subtraction comparison
    if IS_DEBUG or output_image_directory is not None:
        _plot_subtraction_comparison(
            image=image,
            membrane_image=membrane_image,
            subtracted=subtracted,
            subtraction_factor=subtraction_factor,
            output_image_directory=output_image_directory
        )

    return subtracted


def _plot_subtraction_comparison(
    image: torch.Tensor,
    membrane_image: torch.Tensor,
    subtracted: torch.Tensor,
    subtraction_factor: float,
    output_image_directory: os.PathLike | None,
) -> None:
    from matplotlib import pyplot as plt
    fig = plot_subtraction_results(image, membrane_image, subtracted, subtraction_factor)
    if IS_DEBUG:
        plt.show()
    if output_image_directory is not None:
        Path(output_image_directory).mkdir(parents=True, exist_ok=True)
        fname = Path(output_image_directory) / "subtraction_results.png"
        fig.savefig(fname, dpi=300)
    plt.close()

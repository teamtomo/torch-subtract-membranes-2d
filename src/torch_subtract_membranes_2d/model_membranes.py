import logging
from datetime import datetime
import os
from pathlib import Path

import torch
from torch_segment_membranes_2d import predict_membrane_mask

from torch_subtract_membranes_2d.membrane_model import Membrane2D
from torch_subtract_membranes_2d.refine_membrane import refine_membrane
from torch_subtract_membranes_2d.utils.datetime_utils import humanize_timedelta
from torch_subtract_membranes_2d.utils.image_utils import bandpass_filter_image, normalize_2d
from torch_subtract_membranes_2d.trace_paths import trace_paths_in_mask
from torch_subtract_membranes_2d.utils.debug_utils import IS_DEBUG
from torch_subtract_membranes_2d.utils.plotting_utils import (
    plot_preprocessed_image, 
    plot_membrane_mask, 
    plot_paths_on_membrane_mask
)


def model_membranes(
    image: torch.Tensor,
    pixel_spacing_angstroms: float,
    min_path_length_nm: int = 30,
    control_point_spacing_nm: float = 10,
    membrane_mask: torch.Tensor | None = None,
    output_image_directory: os.PathLike | None = None,
) -> list[Membrane2D]:
    # ensure correct input dtypes
    image = image.float()
    if membrane_mask is not None:
        membrane_mask = membrane_mask.to(device=image.device, dtype=torch.bool)

    # normalize and bandpass image
    print("Normalizing and filtering input image...")
    image = normalize_2d(image)
    image = bandpass_filter_image(
        image=image,
        pixel_spacing=pixel_spacing_angstroms,
        highpass_angstroms=300,
        lowpass_angstroms=20,
    )
    print("Input image normalized and filtered")

    if IS_DEBUG or output_image_directory is not None:
        _plot_preprocessed_image(
            image=image,
            output_image_directory=output_image_directory
        )

    # predict membrane segmentation if required
    if membrane_mask is None:
        print("Predicting membrane mask...")
        logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
        membrane_mask = predict_membrane_mask(
            image=image,
            pixel_spacing=pixel_spacing_angstroms,
            probability_threshold=0.8
        )
        membrane_mask = membrane_mask.to(image.device)
        print("Membrane mask predicted")

    if IS_DEBUG or output_image_directory is not None:
        _plot_membrane_mask_debug(membrane_mask, output_image_directory)

    # trace paths for each membrane in mask
    print("Tracing initial paths in membrane mask...")
    paths = trace_paths_in_mask(
        membrane_mask=membrane_mask,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        min_path_length_nm=min_path_length_nm,
        control_point_spacing_nm=control_point_spacing_nm,
    )
    print(f"Traced {len(paths)} initial paths")

    # refine membrane models against image data
    membrane_models: list[Membrane2D] = []
    start = datetime.now()
    for idx, path in enumerate(paths):
        print(f"Refining membrane {idx + 1}/{len(paths)}")
        refined_membrane = refine_membrane(
            path=path,
            image=image,
            pixel_spacing_angstroms=pixel_spacing_angstroms,
            output_image_directory=output_image_directory,
        )
        membrane_models.append(refined_membrane)
    end = datetime.now()
    print(f"time taken for membrane refinement: {humanize_timedelta(end - start)}")

    if IS_DEBUG or output_image_directory is not None:
        _plot_refined_paths_debug(membrane_mask, membrane_models, output_image_directory)


    return membrane_models


def _plot_preprocessed_image(
    image: torch.Tensor,
    output_image_directory: os.PathLike | None,
) -> None:
    from matplotlib import pyplot as plt
    fig = plot_preprocessed_image(image)
    if IS_DEBUG:
        plt.show()
    if output_image_directory is not None:
        Path(output_image_directory).mkdir(parents=True, exist_ok=True)
        fname = Path(output_image_directory) / "preprocessed_image.png"
        fig.savefig(fname, dpi=300)
    plt.close()


def _plot_membrane_mask_debug(
    membrane_mask: torch.Tensor,
    output_image_directory: os.PathLike | None,
) -> None:
    from matplotlib import pyplot as plt
    fig = plot_membrane_mask(membrane_mask)
    if IS_DEBUG:
        plt.show()
    if output_image_directory is not None:
        Path(output_image_directory).mkdir(parents=True, exist_ok=True)
        fname = Path(output_image_directory) / "membrane_mask.png"
        fig.savefig(fname, dpi=300)
    plt.close()


def _plot_refined_paths_debug(
    membrane_mask: torch.Tensor,
    membrane_models: list[Membrane2D],
    output_image_directory: os.PathLike | None,
) -> None:
    from matplotlib import pyplot as plt
    fig = plot_paths_on_membrane_mask(
        membrane_mask=membrane_mask,
        membranes=membrane_models,
        title="refined paths on membrane mask"
    )
    if IS_DEBUG:
        plt.show()
    if output_image_directory is not None:
        Path(output_image_directory).mkdir(parents=True, exist_ok=True)
        fname = Path(output_image_directory) / "refined_paths_on_membrane.png"
        fig.savefig(fname, dpi=300)
    plt.close()


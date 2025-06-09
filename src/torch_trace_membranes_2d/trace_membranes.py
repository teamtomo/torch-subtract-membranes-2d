import torch

from torch_trace_membranes_2d.membrane_model import Membrane2D
from torch_trace_membranes_2d.refine_membrane import refine_membrane
from torch_trace_membranes_2d.utils.image_utils import bandpass_filter_image, normalize_2d
from torch_trace_membranes_2d.utils.skeleton_utils import trace_paths_in_mask


def trace_membranes(
    image: torch.Tensor,
    membrane_mask: torch.Tensor,
    pixel_spacing_angstroms: float,
    min_path_length_nm: int,
    control_point_spacing_nm: float,
) -> list[Membrane2D]:
    # normalize and bandpass image
    image = normalize_2d(image)
    image = bandpass_filter_image(
        image=image,
        pixel_spacing=pixel_spacing_angstroms,
        highpass_angstroms=300,
        lowpass_angstroms=20,
    )

    # trace paths for each membrane in mask
    paths = trace_paths_in_mask(
        membrane_mask=membrane_mask,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        min_path_length_nm=min_path_length_nm,
        control_point_spacing_nm=control_point_spacing_nm,
    )

    # refine membrane models against image data
    membrane_models: list[Membrane2D] = []
    for idx, path in enumerate(paths):
        refined_membrane = refine_membrane(
            path=path,
            image=image,
            pixel_spacing_angstroms=pixel_spacing_angstroms
        )
        membrane_models.append(refined_membrane)

    return membrane_models

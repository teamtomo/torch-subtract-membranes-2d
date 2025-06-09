from unicodedata import normalize

import torch

from torch_trace_membranes_2d.constants import REFINEMENT_HIGHPASS_ANGSTROMS
from torch_trace_membranes_2d.membrane_model import Membrane2D
from torch_trace_membranes_2d.utils.image_utils import bandpass_filter_image, normalize_2d
from torch_trace_membranes_2d.render_membranes import render_membrane_image


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
        image_shape=(h, w)
    )

    # subtract membrane image from bandpassed image
    subtracted = image - (subtraction_factor * membrane_image)

    import mrcfile
    mrcfile.write("image.mrc", image.detach().numpy(), overwrite=True)
    mrcfile.write("membrane_image.mrc", membrane_image.detach().numpy(), overwrite=True)
    mrcfile.write("subtracted.mrc", subtracted.detach().numpy(), overwrite=True)

    return subtracted

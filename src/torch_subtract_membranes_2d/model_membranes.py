import logging

import torch
from torch_segment_membranes_2d import predict_membrane_mask

from torch_subtract_membranes_2d.membrane_model import Membrane2D
from torch_subtract_membranes_2d.refine_membrane import refine_membrane
from torch_subtract_membranes_2d.utils.image_utils import bandpass_filter_image, normalize_2d
from torch_subtract_membranes_2d.utils.skeleton_utils import trace_paths_in_mask
from torch_subtract_membranes_2d.utils.debug_utils import IS_DEBUG, set_matplotlib_resolution


def model_membranes(
    image: torch.Tensor,
    pixel_spacing_angstroms: float,
    min_path_length_nm: int = 30,
    control_point_spacing_nm: float = 10,
    membrane_mask: torch.Tensor | None = None,
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

    if IS_DEBUG:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        ax.set_title("preprocessed image")
        ax.imshow(image.detach().cpu().numpy(), cmap="gray")
        plt.show()

    # predict membrane segmentation if required
    if membrane_mask is None:
        print("Predicting membrane mask...")
        logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
        membrane_mask = predict_membrane_mask(
            image=image,
            pixel_spacing=pixel_spacing_angstroms,
            probability_threshold=0.1
        )
        membrane_mask = membrane_mask.to(image.device)
        print("Membrane mask predicted")

    if IS_DEBUG:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        ax.set_title("membrane mask")
        ax.imshow(membrane_mask.cpu().numpy())
        plt.show()

    # trace paths for each membrane in mask
    print("Tracing paths in membrane mask...")
    paths = trace_paths_in_mask(
        membrane_mask=membrane_mask,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        min_path_length_nm=min_path_length_nm,
        control_point_spacing_nm=control_point_spacing_nm,
    )
    print(f"Traced {len(paths)} membranes")

    # refine membrane models against image data
    membrane_models: list[Membrane2D] = []
    for idx, path in enumerate(paths):
        print(f"Refining membrane {idx + 1}/{len(paths)}")
        refined_membrane = refine_membrane(
            path=path,
            image=image,
            pixel_spacing_angstroms=pixel_spacing_angstroms
        )
        membrane_models.append(refined_membrane)

    if IS_DEBUG:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        ax.set_title("refined paths on membrane mask")
        ax.imshow(membrane_mask.detach().cpu().numpy(), cmap="gray")
        for membrane in membrane_models:
            yx = membrane.path.interpolate(u=torch.linspace(0, 1, steps=100))
            ax.plot(yx[:, -1], yx[:, -2])
        plt.show()

    return membrane_models

if IS_DEBUG:
    set_matplotlib_resolution()
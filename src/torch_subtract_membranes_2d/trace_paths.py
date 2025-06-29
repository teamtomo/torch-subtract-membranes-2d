import os
from pathlib import Path

import skan
import skimage
import torch

from torch_subtract_membranes_2d.path_models.path_2d import Path2D
from torch_subtract_membranes_2d.utils import IS_DEBUG
from torch_subtract_membranes_2d.utils.skeleton_utils import prune_branches, remove_short_paths, \
    skeleton_to_uniformly_spaced_paths
from torch_subtract_membranes_2d.utils.plotting_utils import plot_paths_on_membrane_mask


def trace_paths_in_mask(
    membrane_mask: torch.Tensor,
    min_path_length_nm: float,
    pixel_spacing_angstroms: float,
    control_point_spacing_nm: float,
    output_image_directory: os.PathLike | None = None,
) -> list[Path2D]:
    """Convert a membrane mask to a list of Path2Ds"""
    # skeletonize membrane mask
    skeleton_image = skimage.morphology.skeletonize(membrane_mask.detach().cpu().numpy())

    # prune short branches
    skeleton = skan.Skeleton(skeleton_image)
    skeleton = prune_branches(skeleton)

    # remove short paths from skeleton
    min_path_length_px = (min_path_length_nm * 10) / pixel_spacing_angstroms
    skeleton = remove_short_paths(skeleton, min_length=int(min_path_length_px))

    # construct Path2Ds with target control point spacing from skeleton
    control_point_spacing_px = (control_point_spacing_nm * 10) / pixel_spacing_angstroms
    paths = skeleton_to_uniformly_spaced_paths(
        skeleton=skeleton,
        control_point_spacing=control_point_spacing_px,
        device=membrane_mask.device,
    )

    # remove short paths again
    paths = [
        path
        for path
        in paths
        if path.estimated_length() >= min_path_length_px
    ]

    # ensure closed paths are anticlockwise
    # this ensures images and 1d profiles can be interpreted as
    # inside -> outside from left -> right
    paths = [
        path.as_reversed()
        if path.is_clockwise
        else path
        for path in paths
    ]

    if IS_DEBUG:
        _plot_initial_paths_debug(
            membrane_mask=membrane_mask,
            paths=paths,
            output_image_directory=output_image_directory,
        )

    return paths


def _plot_initial_paths_debug(
    membrane_mask: torch.Tensor,
    paths: list[Path2D],
    output_image_directory: os.PathLike | None,
) -> None:
    from matplotlib import pyplot as plt
    fig = plot_paths_on_membrane_mask(membrane_mask, paths, "initial paths on membrane mask")
    if IS_DEBUG:
        plt.show()
    if output_image_directory is not None:
        fig.savefig(Path(output_image_directory) / "initial_paths.png", dpi=300)
    plt.close()

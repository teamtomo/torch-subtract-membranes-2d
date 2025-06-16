import numpy as np
import skan
import skimage
import torch

from torch_subtract_membranes_2d.path_models.path_2d import Path2D
from torch_subtract_membranes_2d.utils import IS_DEBUG


def prune_branches(skeleton: skan.Skeleton) -> skan.Skeleton:
    """Prune shortest branches skelet"""
    # find junctions (nodes with more than two neighbours in path)
    junction_nodes = np.where(skeleton.degrees > 2)[0]

    # find the shortest connected path for each junction
    paths_to_prune = set()
    for junction in junction_nodes:
        connected_paths = []

        for i in range(skeleton.n_paths):
            path = skeleton.path(i)
            if junction in path:
                path_length = skeleton.path_lengths()[i]
                connected_paths.append((i, path_length))

        if connected_paths:
            shortest_path_idx = min(connected_paths, key=lambda x: x[1])[0]
            paths_to_prune.add(shortest_path_idx)

    # prune the shortest paths
    if paths_to_prune:
        print(list(paths_to_prune))
        pruned_skeleton = skeleton.prune_paths(list(paths_to_prune))
    else:
        pruned_skeleton = skeleton

    return pruned_skeleton


def remove_short_paths(skeleton: skan.Skeleton, min_length: int) -> skan.Skeleton:
    """Remove paths that have fewer pixels than a minimum threshold"""
    paths_to_prune = []

    for i in range(skeleton.n_paths):
        if skeleton.path_lengths()[i] < min_length:
            paths_to_prune.append(i)

    if paths_to_prune:
        return skeleton.prune_paths(paths_to_prune)
    else:
        return skeleton


def skeleton_to_paths(
    skeleton: skan.Skeleton,
    control_point_spacing: float
) -> list[Path2D]:
    """Convert a skeleton into a list of Path2Ds"""
    # we must determine for each path if it is closed or not to construct a path

    # define closure for checking if path is closed
    # condition: a closed path has no ends
    endpoints = skeleton.coordinates[skeleton.degrees == 1]

    def _path_is_closed(i: int):
        path_is_closed = all(
            endpoint not in skeleton.path_coordinates(i)
            for endpoint in endpoints
        )
        return path_is_closed

    # construct path objects
    paths = [
        Path2D(
            control_points=skeleton.path_coordinates(i),
            is_closed=_path_is_closed(i),
            yx_coords=True
        )
        for i in range(skeleton.n_paths)
    ]

    # reconstruct paths with even control point spacing
    paths = [path.as_uniformly_spaced(control_point_spacing) for path in paths]

    return paths


def trace_paths_in_mask(
    membrane_mask: torch.Tensor,
    min_path_length_nm: float,
    pixel_spacing_angstroms: float,
    control_point_spacing_nm: float,
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
    paths = skeleton_to_paths(
        skeleton=skeleton, control_point_spacing=control_point_spacing_px
    )

    # ensure closed paths are anticlockwise
    # this ensures images and 1d profiles can be interpreted as
    # inside -> outside from left -> right
    paths = [
        path.as_reversed()
        if (path.is_clockwise and path.is_closed)
        else path
        for path in paths
    ]

    if IS_DEBUG:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        ax.set_title("initial paths on membrane mask")
        ax.imshow(membrane_mask.detach().cpu().numpy(), cmap="gray")
        for path in paths:
            yx = path.interpolate(u=torch.linspace(0, 1, steps=100))
            ax.plot(yx[:, -1].cpu().numpy(), yx[:, -2].cpu().numpy())
        plt.show()

    return paths
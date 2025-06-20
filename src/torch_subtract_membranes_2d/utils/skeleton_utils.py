import numpy as np
import skan
import torch

from torch_subtract_membranes_2d.path_models.path_2d import Path2D


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


def skeleton_to_uniformly_spaced_paths(
    skeleton: skan.Skeleton,
    control_point_spacing: float,
    device: torch.device,
) -> list[Path2D]:
    """Convert a skeleton into a list of Path2Ds"""
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
            control_points=torch.as_tensor(
                skeleton.path_coordinates(i),
                device=device,
                dtype=torch.float32
            ),
            is_closed=_path_is_closed(i),
            yx_coords=True
        )
        for i in range(skeleton.n_paths)
    ]

    # resample paths with even control point spacing
    paths = [
        path.as_uniformly_spaced(control_point_spacing)
        for path
        in paths
    ]

    return paths



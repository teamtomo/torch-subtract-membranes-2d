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

    # construct path objects
    paths = []
    for path_idx in range(skeleton.n_paths):
        # grab points in path
        path_points = skeleton.path_coordinates(path_idx)

        # check if path is closed
        path_is_closed = _path_is_closed(
            endpoints=endpoints,
            path_coordinates=path_points
        )

        # construct path and append
        control_points = torch.as_tensor(
            path_points,
            device=device,
            dtype=torch.float32
        )
        path = Path2D(
            control_points=control_points,
            is_closed=path_is_closed,
            yx_coords=True
        )
        paths.append(path)

    # resample paths with even control point spacing
    paths = [
        path.as_uniformly_spaced(control_point_spacing)
        for path
        in paths
    ]

    return paths


def _path_is_closed(
    endpoints: np.ndarray,
    path_coordinates: np.ndarray
) -> bool:
    # convert endpoints to list[tuple[int, int]]
    endpoints = [(int(yx[0]), int(yx[1])) for yx in endpoints]

    # convert path coordinates to set[tuple[int, int]]
    path_coordinates = set(
        (int(yx[0]), int(yx[1]))
        for yx
        in path_coordinates
    )

    # if endpoint in path, path is not closed
    if any(endpoint in path_coordinates for endpoint in endpoints):
        return False
    else:
        return True

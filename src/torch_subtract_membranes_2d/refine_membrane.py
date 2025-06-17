import os
from copy import deepcopy
from pathlib import Path

import einops
import torch

from torch_subtract_membranes_2d.membrane_model import Membrane2D
from torch_subtract_membranes_2d.path_models.path_1d import Path1D
from torch_subtract_membranes_2d.path_models.path_2d import Path2D
from torch_subtract_membranes_2d.utils import IS_DEBUG
from torch_subtract_membranes_2d.utils.debug_utils import set_matplotlib_resolution
from torch_subtract_membranes_2d.utils.image_utils import smooth_tophat_1d
from torch_subtract_membranes_2d.utils.path_utils import sample_image_along_path
from torch_subtract_membranes_2d.constants import MEMBRANE_BILAYER_WIDTH_ANGSTROMS


def refine_membrane(
    path: Path2D,
    image: torch.Tensor,
    pixel_spacing_angstroms: float,
    n_iterations: int = 500,
    debug_output_directory: os.PathLike | None = None,
) -> Membrane2D:
    # get device
    device = image.device

    # grab original control points
    original_control_points = path.control_points.clone()
    n_control_points = len(original_control_points)

    # grab normals to path at initial control point positions
    u_at_control_points = path.get_closest_u(query_points=original_control_points)
    normals_at_control_points = path.get_normals(u=u_at_control_points).to(device)

    # set up distances from path for each sample perpendicular to path
    maximum_distance = (2 * MEMBRANE_BILAYER_WIDTH_ANGSTROMS) / pixel_spacing_angstroms
    maximum_distance = int(maximum_distance)
    perpendicular_steps = torch.arange(-1 * maximum_distance, maximum_distance + 1, step=1)

    # define membranogram dimensions
    membranogram_h, membranogram_w = int(path.estimated_length()), len(perpendicular_steps)

    # set up weights as a function of perpendicular distance to the path
    weights_1d = smooth_tophat_1d(
        length=len(perpendicular_steps),
        center_width=len(perpendicular_steps) // 3,
        rolloff_width=len(perpendicular_steps) // 5,
    ).to(device)

    # setup parameters to be optimized
    perpendicular_shifts_nanometers = torch.zeros(size=(n_control_points,), requires_grad=True, device=device)
    signal_scale_control_points = torch.ones(size=(n_control_points,), requires_grad=True, device=device)

    # setup optimizer
    params = [
        perpendicular_shifts_nanometers,
        signal_scale_control_points,
    ]
    optimizer = torch.optim.Adam(
        params=params,
        lr=0.01,
    )

    # Initialize loss tracking
    min_loss = float('inf')
    initial_loss = float('-inf')
    recent_losses = []

    # start refinement
    for i in range(n_iterations):
        # zero gradients
        optimizer.zero_grad()

        ## update control points
        # first scale to pixels
        perpendicular_shifts_px = (perpendicular_shifts_nanometers * 10) / pixel_spacing_angstroms

        # subtract the mean from shifts to prevent membranes "flying away"
        perpendicular_shifts_px = perpendicular_shifts_px - torch.mean(perpendicular_shifts_px)

        # move control points along path normal vectors
        # perpendicular shifts: (n_control_points, )
        # normals: (n_control_points, 2)
        perpendicular_shifts_px = einops.repeat(perpendicular_shifts_px, "b -> b 2")
        shifts = perpendicular_shifts_px * normals_at_control_points
        shifts = shifts.to(device)
        updated_control_points = original_control_points + shifts

        # construct Path2D from updated control points
        path = Path2D(control_points=updated_control_points, is_closed=path.is_closed, yx_coords=True)

        # sample perpendicular segments along path with updated control points
        membranogram, out_of_bounds_mask = sample_image_along_path(
            path=path,
            image=image,
            distances=perpendicular_steps,
            n_samples=membranogram_h
        )

        # make 1d average
        average_1d = torch.mean(membranogram, dim=0)

        # reweight 1d average according to signal scale estimate for loss calc.
        signal_scale_spline = Path1D(
            control_points=signal_scale_control_points, is_closed=path.is_closed
        )
        signal_scale_estimates = signal_scale_spline.interpolate(
            u=torch.linspace(0, 1, steps=membranogram_h)
        )
        signal_scale_estimates = einops.repeat(
            signal_scale_estimates, "h -> h w", w=membranogram_w
        )
        average_2d_scaled = average_1d * signal_scale_estimates
        average_2d_scaled = average_2d_scaled * out_of_bounds_mask

        if i == 0:
            original_membranogram = membranogram.clone().detach().cpu().numpy()
            original_average_2d_scaled = average_2d_scaled.clone().detach().cpu().numpy()

        # calculate loss
        weighted_mse = torch.mean(weights_1d * (membranogram - average_2d_scaled) ** 2)
        loss = weighted_mse

        # backprop and step optimizer
        loss.backward()
        optimizer.step()

        # Update loss tracking
        current_loss = loss.item()
        min_loss = min(min_loss, current_loss)
        if i == 0:
            initial_loss = deepcopy(current_loss)

        recent_losses.append(current_loss)
        if len(recent_losses) > 10:
            recent_losses.pop(0)

        # Calculate average difference if we have enough data
        avg_diff = 0
        if len(recent_losses) >= 2:
            diffs = [abs(recent_losses[i] - recent_losses[i - 1]) for i in range(1, len(recent_losses))]
            avg_diff = sum(diffs) / len(diffs)

        # logging
        progress_msg = f"{i + 1:4d}/{n_iterations} | loss: {current_loss:.6f} (initial: {initial_loss:.6f} min: {min_loss:.6f} delta: {avg_diff:.1e})"
        if i == n_iterations - 1:
            print(f"\r{progress_msg}")
        else:
            print(f"\r{progress_msg}", end="", flush=True)

    if IS_DEBUG or debug_output_directory is not None:
        from matplotlib import pyplot as plt
        plt.show()
        fig, ax = plt.subplots(ncols=4)
        ax[0].set_title("initial\ndata")
        ax[0].imshow(original_membranogram, cmap="gray")
        ax[1].set_title("initial\nreconstruction")
        ax[1].imshow(original_average_2d_scaled, cmap="gray")
        ax[2].set_title("refined\ndata")
        ax[2].imshow(membranogram.detach().cpu().numpy(), cmap="gray")
        ax[3].set_title("refined\nreconstruction")
        ax[3].imshow(average_2d_scaled.detach().cpu().numpy(), cmap="gray")
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        ax[3].axis('off')
        fig.tight_layout()
        plt.show()
        if debug_output_directory is not None:
            Path(debug_output_directory).mkdir(parents=True, exist_ok=True)
            import random
            import string
            random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
            fname = Path(debug_output_directory) / f"membrane_refinement_{random_string}.png"
            fig.savefig(fname, dpi=300)
        plt.close()

    return Membrane2D(
        profile_1d=average_1d.detach(),
        weights_1d=weights_1d.detach(),
        path_control_points=path.control_points.detach(),
        signal_scale_control_points=signal_scale_control_points.detach(),
        path_is_closed=path.is_closed,
    )

if IS_DEBUG:
    set_matplotlib_resolution()
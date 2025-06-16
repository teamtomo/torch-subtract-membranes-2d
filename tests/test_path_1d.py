import numpy as np
import pytest
import torch

from torch_subtract_membranes_2d.path_models.path_1d import Path1D

# parameter for cuda device tests which skips test if cuda not available
cuda_device_parameter = pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
))

@pytest.mark.parametrize("device", ["cpu", cuda_device_parameter])
def test_open_path_interpolation(device):
    # simple line from [0, 0] to [3, 0]
    control_points = np.array([0, 1, 2, 3])
    control_points = torch.tensor(control_points, device=device, dtype=torch.float32)
    path = Path1D(control_points=control_points, is_closed=False)

    u0 = torch.tensor([0]).float()
    interp_u0 = path.interpolate(u0)
    assert torch.allclose(interp_u0, torch.tensor([0.0, ], device=device), atol=1e-5)

    u0_5 = torch.tensor([0.5]).float()
    interp_u0_5 = path.interpolate(u0_5)
    assert torch.allclose(interp_u0_5, torch.tensor([1.5, ], device=device), atol=1e-5)

    u1 = torch.tensor([1]).float()
    interp_u1 = path.interpolate(u1)
    assert torch.allclose(interp_u1, torch.tensor([3.0, ], device=device), atol=1e-5)

@pytest.mark.parametrize("device", ["cpu", cuda_device_parameter])
def test_closed_path_interpolation(device):
    # closed CW path with xy coordinates
    control_points = np.array([0, 1, 2, 3, 2, 1])
    control_points = torch.tensor(control_points, device=device, dtype=torch.float32)
    path = Path1D(control_points=control_points, is_closed=True)

    # start point should be the same as end point
    u0 = torch.tensor([0]).float()
    u1 = torch.tensor([1]).float()
    interp_u0 = path.interpolate(u0)
    interp_u1 = path.interpolate(u1)
    assert torch.allclose(interp_u0, interp_u1, atol=1e-5)
    assert torch.allclose(interp_u0, torch.tensor([0.], device=device), atol=1e-5)

    # should be 3 in middle...
    u0_5 = torch.tensor([0.5]).float()
    interp_u0_5 = path.interpolate(u0_5)
    assert torch.allclose(interp_u0_5, torch.tensor([3.0, ], device=device), atol=1e-5)

@pytest.mark.parametrize("device", ["cpu", cuda_device_parameter])
def test_optimization(device):
    # setup some initial control points
    initial_control_points = torch.zeros(size=(30, ), device=device)
    path = Path1D(control_points=initial_control_points, is_closed=False)
    path.control_points.requires_grad_(True)

    # target function
    def f(x):
        return torch.sin(x * 8 * torch.pi)

    # optimizer setup
    optimizer = torch.optim.Adam(params=[path.control_points], lr=1e-2)

    # optimize such that interpolating over the [0, 1] interval gives 2d points where x=u, y=f(x)
    for i in range(1000):
        # zero gradients
        optimizer.zero_grad()

        # random samples
        u = torch.rand(size=(100,), device=device)
        ground_truth_x = u
        ground_truth_y = f(ground_truth_x)

        # calculate mean squared error
        predicted_y = path.interpolate(u=u)
        mse = torch.mean((predicted_y - ground_truth_y) ** 2)

        # backprop
        mse.backward()
        optimizer.step()

    assert mse.item() < 1e-4

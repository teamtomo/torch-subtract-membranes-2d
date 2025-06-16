import einops
import numpy as np
import pytest
import torch

from torch_subtract_membranes_2d.path_models.path_2d import Path2D

# parameter for cuda device tests which skips test if cuda not available
cuda_device_parameter = pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
))


@pytest.mark.parametrize(
    "control_points, closed, yx_coords, expected",
    [
        # closed CW path with xy coordinates
        (np.array([[0, 0], [0, 10], [10, 10], [10, 0]]), True, False, True),

        # closed CCW path with xy coordinates
        (np.array([[0, 0], [10, 0], [10, 10], [0, 10]]), True, False, False),

        # closed CW path with yx coordinates
        (np.array([[0, 0], [10, 0], [10, 10], [0, 10]]), True, True, True),

        # closed CCW path with yx coordinates
        (np.array([[0, 0], [0, 10], [10, 10], [10, 0]]), True, True, False),
    ]
)
def test_is_clockwise(
        control_points: np.ndarray,
        closed: bool,
        yx_coords: bool,
        expected: bool,
):
    control_points = torch.tensor(control_points, dtype=torch.float32)
    path = Path2D(control_points=control_points, is_closed=closed, yx_coords=yx_coords)
    assert path.is_clockwise is expected


@pytest.mark.parametrize("device", ["cpu", cuda_device_parameter])
def test_open_path_interpolation(device):
    # simple line from [0, 0] to [3, 0]
    control_points = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    control_points = torch.tensor(control_points, device=device, dtype=torch.float32)
    path = Path2D(control_points=control_points, is_closed=False)

    u0 = torch.tensor([0], device=device).float()
    interp_u0 = path.interpolate(u0)
    if device != "meta":
        assert torch.allclose(interp_u0, torch.tensor([0.0, 0.0], device=device), atol=1e-5)

    u0_5 = torch.tensor([0.5], device=device).float()
    interp_u0_5 = path.interpolate(u0_5)
    if device != "meta":
        assert torch.allclose(interp_u0_5, torch.tensor([1.5, 0.0], device=device), atol=1e-5)

    u1 = torch.tensor([1], device=device).float()
    interp_u1 = path.interpolate(u1)
    if device != "meta":
        assert torch.allclose(interp_u1, torch.tensor([3.0, 0.0], device=device), atol=1e-5)


@pytest.mark.parametrize("device", ["cpu", cuda_device_parameter])
def test_closed_path_interpolation(device):
    # closed CW path with xy coordinates
    control_points = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
    control_points = torch.tensor(control_points, device=device, dtype=torch.float32)
    path = Path2D(control_points=control_points, is_closed=True)

    # start point should be the same as end point
    u0 = torch.tensor([0]).float()
    u1 = torch.tensor([1]).float()
    interp_u0 = path.interpolate(u0)
    interp_u1 = path.interpolate(u1)
    if device != "meta":
        assert torch.allclose(interp_u0, interp_u1, atol=1e-5)


@pytest.mark.parametrize("device", ["cpu", cuda_device_parameter])
def test_get_tangents(device):
    # simple line from [0, 0] to [3, 0]
    control_points = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    control_points = torch.tensor(control_points, device=device, dtype=torch.float32)
    path = Path2D(control_points=control_points, is_closed=False)

    u0 = torch.tensor([0]).float()
    tangent_u0 = path.get_tangents(u0)
    if device != "meta":
        assert torch.allclose(tangent_u0, torch.tensor([1.0, 0.0], device=device))

    u0_5 = torch.tensor([0.5]).float()
    tangent_u0_5 = path.get_tangents(u0_5)
    if device != "meta":
        assert torch.allclose(tangent_u0_5, torch.tensor([1.0, 0.0], device=device))

    u1 = torch.tensor([1]).float()
    tangent_u1 = path.get_tangents(u1)
    if device != "meta":
        assert torch.allclose(tangent_u1, torch.tensor([1.0, 0.0], device=device))


@pytest.mark.parametrize("device", ["cpu", cuda_device_parameter])
def test_get_normals(device):
    # simple line from [0, 0] to [3, 0]
    control_points = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    control_points = torch.tensor(control_points, device=device, dtype=torch.float32)
    path = Path2D(control_points=control_points, is_closed=False)

    u0 = torch.tensor([0]).float()
    normal_u0 = path.get_normals(u0)
    if device != "meta":
        assert torch.allclose(normal_u0, torch.tensor([0.0, -1.0], device=device))

    u0_5 = torch.tensor([0.5]).float()
    normal_u0_5 = path.get_normals(u0_5)
    if device != "meta":
        assert torch.allclose(normal_u0_5, torch.tensor([0.0, -1.0], device=device))

    u1 = torch.tensor([1]).float()
    normal_u1 = path.get_normals(u1)
    if device != "meta":
        assert torch.allclose(normal_u1, torch.tensor([0.0, -1.0], device=device))


@pytest.mark.parametrize("device", ["cpu", cuda_device_parameter])
def test_get_normals_yx(device):
    # simple line from [0, 0] to [3, 0]
    control_points = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])
    control_points = torch.tensor(control_points, device=device, dtype=torch.float32)
    path = Path2D(control_points=control_points, is_closed=False, yx_coords=True)

    u0 = torch.tensor([0]).float()
    normal_u0 = path.get_normals(u0)
    if device != "meta":
        assert torch.allclose(normal_u0, torch.tensor([-1.0, 0.0], device=device))

    u0_5 = torch.tensor([0.5]).float()
    normal_u0_5 = path.get_normals(u0_5)
    if device != "meta":
        assert torch.allclose(normal_u0_5, torch.tensor([-1.0, 0.0], device=device))

    u1 = torch.tensor([1]).float()
    normal_u1 = path.get_normals(u1)
    if device != "meta":
        assert torch.allclose(normal_u1, torch.tensor([-1.0, 0.0], device=device))


@pytest.mark.parametrize("device", ["cpu", cuda_device_parameter])
def test_get_closest_u(device):
    # simple line from [0, 0] to [3, 0]
    control_points = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])
    control_points = torch.tensor(control_points, device=device, dtype=torch.float32)
    path = Path2D(control_points=control_points, is_closed=False)

    # generate 100 points all 10 units away from the line
    xy = np.linspace(start=(10, 0), stop=(10, 3), num=100)
    xy = torch.as_tensor(xy, device=device, dtype=torch.float32)

    # find closest u value and distance
    closest_u = path.get_closest_u(xy)
    closest_points = path.interpolate(closest_u)
    refined_distances = torch.linalg.norm(xy - closest_points, dim=-1)

    # check that refined distances match expected distances
    expected_distances = torch.ones(size=(len(xy),), device=device) * 10
    if device != "meta":
        assert torch.allclose(refined_distances, expected_distances, atol=1e-4)

    # check that u values match expectations
    expected_u = torch.linspace(0, 1, steps=100, device=device)
    if device != "meta":
        assert torch.allclose(closest_u, expected_u, atol=1e-3)


@pytest.mark.parametrize("device", ["cpu", cuda_device_parameter])
def test_as_uniformly_spaced_open_path(device):
    # simple line from [0, 0] to [0, 4]
    control_points = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]])
    control_points = torch.tensor(control_points, device=device, dtype=torch.float32)
    path = Path2D(control_points=control_points, is_closed=False)

    new_path = path.as_uniformly_spaced(spacing=0.1)
    diffs = torch.diff(new_path.control_points, dim=0)
    if device != "meta":
        assert torch.allclose(diffs, torch.tensor([0.0, 0.1], device=device), atol=1e-2)


@pytest.mark.parametrize("device", ["cpu", cuda_device_parameter])
def test_as_uniformly_space_closed_path(device):
    # closed CW path with xy coordinates
    control_points = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
    control_points = torch.tensor(control_points, device=device, dtype=torch.float32)
    path = Path2D(control_points=control_points, is_closed=True)

    new_path = path.as_uniformly_spaced(spacing=0.1)
    diffs = torch.diff(new_path.control_points, dim=0)
    lengths = torch.linalg.norm(diffs, dim=-1)
    if device != "meta":
        assert torch.allclose(lengths, torch.tensor([0.1], device=device), atol=1e-2)


@pytest.mark.parametrize("device", ["cpu", cuda_device_parameter])
def test_optimization(device):
    # setup some initial 2d control points
    control_points = torch.zeros(size=(30, 2), device=device, dtype=torch.float32)
    path = Path2D(control_points=control_points, is_closed=False)
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
        u = torch.rand(size=(100,), device=device, dtype=torch.float32)
        ground_truth_x = u
        ground_truth_y = f(ground_truth_x)

        # calculate mean squared error
        interpolated = path.interpolate(u=u)
        predicted_x, predicted_y = einops.rearrange(interpolated, "b xy -> xy b")
        mse = torch.mean((predicted_x - ground_truth_x) ** 2 + (predicted_y - ground_truth_y) ** 2)

        # backprop
        mse.backward()
        optimizer.step()

    assert mse.item() < 1e-4

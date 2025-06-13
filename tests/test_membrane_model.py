import numpy as np

from torch_subtract_membranes_2d import save_membranes, load_membranes
from torch_subtract_membranes_2d.membrane_model import Membrane2D

def test_membrane_model_list_save_load_roundtrip(tmp_path):
    membrane = Membrane2D(
        profile_1d=np.random.random(10),
        weights_1d=np.random.random(10),
        path_control_points=np.random.random((10, 2)),
        signal_scale_control_points=np.random.random(10),
        path_is_closed=True
    )
    membranes = [membrane, membrane]
    tmp_file = tmp_path / "tmp.json"
    save_membranes(membranes, tmp_file)
    loaded_membranes = load_membranes(tmp_file)
    for old, new in zip(membranes, loaded_membranes):
        assert old == new





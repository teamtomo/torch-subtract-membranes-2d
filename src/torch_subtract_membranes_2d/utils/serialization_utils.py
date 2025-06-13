import os
from typing import List

from pydantic import TypeAdapter

from torch_subtract_membranes_2d.membrane_model import Membrane2D

# adapter class to help with serialization/deserialization
MembraneList = TypeAdapter(List[Membrane2D])


def save_membranes(
    membranes: List[Membrane2D], path: os.PathLike
) -> None:
    """Save membranes to json."""
    json = MembraneList.dump_json(membranes, indent=2)
    with open(path, "wb") as f:
        f.write(json)
    return


def load_membranes(path: os.PathLike) -> List[Membrane2D]:
    """Load membranes from json."""
    with open(path, "rb") as f:
        json = f.read()
    membranes = MembraneList.validate_json(json)
    return membranes

import enum
from dataclasses import dataclass
from typing import Any, Iterable

import typeguard
import numpy as np


class Type(enum.Enum):
    STRUCTURE = 'structure'
    PES = 'pes'
    PROPERTY = 'property'
    METADATA = 'metadata'
    CUSTOM = 'custom'


@dataclass(frozen=True)
class Quantity:
    name: str
    type: Type
    shape: tuple[int, ...]
    dtype: type = np.float64
    per_atom: bool = False
    default: Any = np.nan


# structural
positions = Quantity('positions', Type.STRUCTURE, (3,), per_atom=True)
numbers = Quantity('numbers', Type.STRUCTURE, (1,), np.uint8, True, 0)
cell = Quantity('cell', Type.STRUCTURE, (3, 3), default=0)

# pes
energy = Quantity('energy', Type.PES, (1,), default=None)
forces = Quantity('forces', Type.PES, (3,), per_atom=True)
stress = Quantity('stress', Type.PES, (3, 3), default=None)

# properties
per_atom_energy = Quantity('per_atom_energy', Type.PROPERTY, (1,))

# metadata
identifier = Quantity('identifier', Type.METADATA, (1,), int, default=-1)
stdout = Quantity('stdout', Type.METADATA, (1,), str, default='')


@dataclass
@typeguard.typechecked
class Quantities:
    """Just a container"""
    per_atom = (numbers, positions, forces)
    regular = (cell, energy, stress)
    properties = (per_atom_energy,)
    meta = (identifier, stdout)

    resettable = {forces, energy, stress}
    write_to_header = {energy, stress}
    per_atom_dtypes: np.dtype = None

    def __post_init__(self):
        self.default = self.per_atom + self.regular + self.properties + self.meta
        self.all = {q.name: q for q in self.default}
        self.per_atom_dtypes = self.get_per_atom_dtype()

    def get_per_atom_dtype(self) -> np.dtype:
        """"""
        return np.dtype([(q.name, q.dtype, q.shape) for q in self.per_atom])


def register_quantity(
    name: str,
    shape: tuple[int, ...],
    type: Type = Type.CUSTOM,
    dtype: type = np.float64,
    per_atom: bool = False,
    default: Any = np.nan
) -> None:
    """"""
    if per_atom:
        assert len(shape) == 1
    quantity = Quantity(name, type, shape, dtype, per_atom, default)
    if per_atom:
        quantities.per_atom += (quantity,)
        quantities.per_atom_dtypes = quantities.get_per_atom_dtype()
    else:
        quantities.regular += (quantity,)
        quantities.write_to_header.add(quantity)

    quantities.all[quantity.name] = quantity
    quantities.resettable.add(quantity)


quantities = Quantities()


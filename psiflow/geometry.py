from __future__ import annotations  # necessary for type-guarding class methods

import io
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union, Sequence, Any

import numpy as np
import typeguard
from ase import Atoms
from ase.data import atomic_masses, chemical_symbols, atomic_numbers
from ase.io.extxyz import key_val_dict_to_str, key_val_str_to_dict_regex
from parsl.app.app import python_app

import psiflow
from psiflow.quantities import quantities, Quantity


chemical_symbols = np.array(chemical_symbols)


@typeguard.typechecked
class Geometry:
    """
    Represents an atomic structure with associated properties.

    This class encapsulates the atomic structure, including atom positions, cell parameters,
    and various physical properties such as energy and forces.

    Attributes:
        per_atom (np.recarray): Record array containing per-atom properties.
        cell (np.ndarray): 3x3 array representing the unit cell vectors.
        energy (Optional[float]): Total energy of the system.
        stress (Optional[np.ndarray]): Stress tensor of the system.
        metadata (dict): Dictionary to store meta information.

    Additional quantities can be stored through constructor kwargs
    """
    per_atom: np.recarray
    cell: np.ndarray
    energy: Optional[float]
    stress: Optional[np.ndarray]
    metadata: dict

    def __init__(
        self,
        per_atom: np.recarray,
        cell: np.ndarray,
        energy: Optional[float] = None,
        stress: Optional[np.ndarray] = None,
        metadata: Optional[dict] = None,
        **kwargs
    ):
        """
        Initialize a Geometry instance, though the preferred way of instantiating
        proceeds via the `from_data` or `from_atoms` class methods
        """
        self.per_atom = per_atom.astype(quantities.per_atom_dtypes)     # copies data
        # TODO: why do we use a [0]-cell instead of None for nonperiodic structures?
        self.cell = cell.astype(np.float64)
        assert self.cell.shape == (3, 3)
        self.energy = energy
        self.stress = stress
        self.metadata = metadata or {}

        # TODO: this warning needs to be more clear?
        # TODO: initialise all quantities with default values already?
        for key, value in kwargs.items():
            if key not in quantities.all:
                warnings.warn(f'Psiflow does not know the Quantity "{key}". This will lead to problems..')
            setattr(self, key, value)

        # TODO: temporary measure to ease the transition
        self.order = self.metadata

    def __getitem__(self, item: str):
        return self.metadata[item]  # TODO: access metadata as dict makes sense?

    def __setitem__(self, item: str, value) -> None:
        self.metadata[item] = value

    def reset(self) -> None:
        """
        Reset all computed properties of the geometry to their default values.
        """
        for q in quantities.resettable:
            if q.per_atom:
                self.per_atom[q.name] = q.default
            else:
                setattr(self, q.name, q.default)

    def clean(self) -> None:
        """
        Clean the geometry by resetting properties and removing additional information.
        """
        self.reset()
        self.metadata = {}

    def __eq__(self, other) -> bool:
        """
        Check if two Geometry instances are structurally equal.

        Args:
            other: The other object to compare with.

        Returns:
            bool: True if the geometries are equal, False otherwise.
        """
        if (
            isinstance(other, Geometry) and
            len(self) == len(other) and
            self.periodic == other.periodic and
            np.allclose(self.per_atom.numbers, other.per_atom.numbers) and
            np.allclose(self.per_atom.positions, other.per_atom.positions) and
            np.allclose(self.cell, other.cell)
        ):
            return True
        return False

    def align_axes(self) -> None:
        """
        Align the axes of the unit cell to a canonical representation for periodic systems.
        """
        if self.periodic:  # only do something if periodic:
            positions = self.per_atom.positions
            cell = self.cell
            transform_lower_triangular(positions, cell, reorder=False)
            reduce_box_vectors(cell)

    def __len__(self):
        """
        Get the number of atoms in the geometry.

        Returns:
            int: The number of atoms.
        """
        return len(self.per_atom)

    def to_string(self) -> str:
        """
        Convert the Geometry instance to a string representation in extended XYZ format.

        Returns:
            str: String representation of the geometry.
        """
        if self.periodic:
            cell = self.cell.T      # transpose because key_val_dict_to_str flattens in Fortran order
            lattice_str = 'Lattice="{}" pbc="T T T"'.format(
                ' '.join([f'{x:.8f}' for x in np.reshape(cell, 9, order='F')])
            )
        else:
            lattice_str = 'pbc="F F F" '

        write_property, write_fmts = [], ['%-2s'] + ['%16.8f'] * 3
        properties_str = "Properties=species:S:1:pos:R:3"
        for q in quantities.per_atom[2:]:
            write = not np.any(np.isnan(self.per_atom[q.name]))
            if write:
                property_str, fmt = _quantity_to_extxyz(q)
                write_property.append(q)
                properties_str += property_str
                write_fmts.extend([fmt] * q.shape[0])

        values_dict = (
            {q.name: getattr(self, q.name, q.default) for q in quantities.write_to_header} |
            _format_metadata(self.metadata)
        )
        key_val_str = key_val_dict_to_str({k: v for k, v in values_dict.items() if _check_value(v)})
        header = f'{lattice_str} {properties_str} {key_val_str}'

        symbols = chemical_symbols[self.per_atom.numbers]
        data = [symbols, self.per_atom.positions]
        for q in write_property:
            data.append(self.per_atom[q.name])
        arr = np.concatenate(data, axis=1, dtype='O')
        s = io.BytesIO()
        np.savetxt(s, arr, fmt=' '.join(write_fmts))
        body = s.getvalue().decode('UTF-8')

        return f'{len(self)}\n{header}\n{body}'

    def save(self, path_xyz: Union[Path, str]) -> None:
        """
        Save the Geometry instance to an XYZ file.

        Args:
            path_xyz (Union[Path, str]): Path to save the XYZ file.
        """
        path_xyz = psiflow.resolve_and_check(Path(path_xyz))
        with open(path_xyz, "w") as f:
            f.write(self.to_string())

    def copy(self) -> Geometry:
        """
        Create a deep copy of the Geometry instance.

        Returns:
            Geometry: A new Geometry instance with the same data.
        """
        return Geometry.from_string(self.to_string())

    @classmethod
    def from_string(cls, s: str, natoms: Optional[int] = None) -> Optional[GeometryLike]:
        """
        Create a Geometry instance from a string representation in extended XYZ format.

        Args:
            s (str): String representation of the geometry.
            natoms (Optional[int], optional): Number of atoms (if known). Defaults to None.

        Returns:
            Optional[Geometry]: A new Geometry instance, or None if the string is empty.
        """
        if len(s) == 0:
            return None
        if natoms:
            # TODO: what does this do? Remove natoms arg?
            # i-PI nonperiodic starts with empty -> rstrip!
            print('Geometry.from_string() weirdness')
            s = s.strip()

        natoms_str, header, body = s.split('\n', 2)
        comment_dict = key_val_str_to_dict_regex(header)

        if header.startswith(NullState.key):
            null = NullState(_extract_metadata(comment_dict))
            null.metadata.update(**_extract_metadata(comment_dict))
            return null

        # read and format per_atom data
        # first 4 columns are always occupied by symbols and positions
        per_atom_keys = {'numbers': slice(0, 1), 'positions': slice(1, 4)}
        if "Properties" in comment_dict:
            # TODO: this has to be true, otherwise it was not extxyz
            properties = comment_dict.pop('Properties').split(":")[6:]
            count = 4
            for i in range(len(properties) // 3):
                name = properties[3 * i]
                ncolumns = int(properties[3 * i + 2])
                per_atom_keys[name] = slice(count, count + ncolumns)
                count += ncolumns

        natoms = int(natoms_str)
        data = body.split()
        assert len(data) % natoms == 0, 'String is not in proper extxyz format'
        width = len(data) // natoms
        data[::width] = [atomic_numbers[symbol] for symbol in data[::width]]
        arr = np.array(data, dtype=float).reshape(-1, width)
        per_atom = np.recarray(natoms, dtype=quantities.per_atom_dtypes)
        for q in quantities.per_atom:
            if q.name in per_atom_keys:
                columns = per_atom_keys[q.name]
                per_atom[q.name] = arr[:, columns]
            else:
                per_atom[q.name] = q.default

        # transpose cell to undo key_val_str_to_dict_regex Fortran ordering
        kwargs = {'cell': comment_dict.pop('Lattice', np.zeros((3, 3))).T}
        metadata = _extract_metadata(comment_dict)
        for q in quantities.write_to_header:
            if q.name not in metadata:
                kwargs[q.name] = comment_dict.pop(q.name, q.default)

        return cls(per_atom=per_atom, metadata=metadata, **kwargs)

    @classmethod
    def load(cls, path_xyz: Union[Path, str]) -> Geometry:
        """
        Load a Geometry instance from an XYZ file.

        Args:
            path_xyz (Union[Path, str]): Path to the XYZ file.

        Returns:
            Geometry: A new Geometry instance loaded from the file.
        """
        path_xyz = psiflow.resolve_and_check(Path(path_xyz))
        assert path_xyz.exists()
        with open(path_xyz, "r") as f:
            content = f.read()
        return cls.from_string(content)

    @property
    def periodic(self) -> bool:
        """
        Check if the geometry is periodic.

        Returns:
            bool: True if the geometry is periodic, False otherwise.
        """
        return bool(np.any(self.cell))

    @property
    def per_atom_energy(self) -> Optional[float]:
        """
        Calculate the energy per atom.

        Returns:
            Optional[float]: Energy per atom if total energy is available, None otherwise.
        """
        if self.energy is None:
            return None
        else:
            return self.energy / len(self)

    @property
    def volume(self) -> Optional[float]:
        """
        Calculate the volume of the unit cell.

        Returns:
            float: Volume of the unit cell for periodic systems, None for non-periodic systems.
        """
        if not self.periodic:
            return None
        else:
            return np.linalg.det(self.cell)
        
    @property
    def atomic_masses(self) -> np.ndarray[float]:
        """
        Get the atomic masses of the atoms in the geometry.

        Returns:
         np.ndarray: Array of atomic masses.
        """
        return atomic_masses[self.numbers]

    @property
    def numbers(self) -> np.ndarray[int]:
        """Return atomic numbers as a flattened array."""
        return self.per_atom.numbers.flatten()

    @classmethod
    def from_data(
        cls,
        numbers: np.ndarray,
        positions: np.ndarray,
        cell: Optional[np.ndarray],
    ) -> Geometry:
        """
        Create a Geometry instance from atomic numbers, positions, and cell data.

        Args:
            numbers (np.ndarray): Array of atomic numbers.
            positions (np.ndarray): Array of atomic positions.
            cell (Optional[np.ndarray]): Unit cell vectors (or None for non-periodic systems).

        Returns:
            Geometry: A new Geometry instance.
        """
        per_atom = np.recarray(len(numbers), dtype=quantities.per_atom_dtypes)
        per_atom.numbers[:] = numbers.reshape(-1, 1)
        per_atom.positions[:] = positions
        for q in quantities.per_atom[2:]:
            per_atom[q.name] = q.default
        cell = cell.copy() if cell is not None else np.zeros((3, 3))
        return Geometry(per_atom, cell)

    @classmethod
    def from_atoms(cls, atoms: Atoms) -> Geometry:
        """
        Create a Geometry instance from an ASE Atoms object.

        Args:
            atoms (Atoms): ASE Atoms object.

        Returns:
            Geometry: A new Geometry instance.
        """
        # ASE is idiotic here
        data = {k: v for k, v in atoms.arrays.items()} | atoms.info
        if atoms.calc is not None:
            data |= atoms.calc.results

        geometry = cls.from_data(
            data.pop('numbers'),
            data.pop('positions'),
            atoms.cell.array if np.any(atoms.pbc) else None
        )
        for q in [q for q in quantities.per_atom[2:] if q.name in data]:
            geometry.per_atom[q.name] = data.pop(q.name)
        for q in quantities.write_to_header:
            value = data.pop(q.name, q.default)
            setattr(geometry, q.name, value)
        geometry.metadata.update(**_extract_metadata(data))
        return geometry


def _check_value(value) -> bool:
    if (
        value is None or
        isinstance(value, float) and np.isnan(value) or
        (isinstance(value, np.ndarray) and np.all(np.isnan(value)))
    ):
        return False
    return True


def _quantity_to_extxyz(quantity: Quantity) -> tuple[str, str]:
    if np.issubdtype(quantity.dtype, np.integer) or np.issubdtype(quantity.dtype, np.bool_):
        # ASE replaces logical values with 'T' and 'F' -> treat as int
        token, fmt = 'I', '%8d'
    elif np.issubdtype(quantity.dtype, np.floating):
        token, fmt = 'R', '%16.8f'
    elif np.issubdtype(quantity.dtype, np.character):
        token, fmt = 'S', '%16s'
    # elif np.issubdtype(quantity.dtype, np.bool_):
    #     token, fmt = 'L', '%8d'
    else:
        raise NotImplementedError
    return f":{quantity.name}:{token}:{quantity.shape[0]}", fmt


def _extract_metadata(data: dict[str, Any]) -> dict[str, Any]:
    return {k[5:]: v for k, v in data.items() if k.startswith('META_')}


def _format_metadata(data: dict[str, Any]) -> dict[str, Any]:
    return {f'META_{k}': v for k, v in data.items()}


@dataclass(frozen=True)     # immutable
class NullState:
    """
    Dummy geometry placeholder that pops up when things went wrong.

    Attributes:
        metadata (dict): Dictionary to store meta information.

    """
    key = 'NULLSTATE'
    metadata: dict = field(default_factory=lambda: {})

    def __eq__(self, other) -> bool:
        return isinstance(other, NullState)

    def __getitem__(self, item: str):
        return self.metadata[item]

    def __setitem__(self, item: str, value) -> None:
        self.metadata[item] = value

    def to_string(self) -> str:
        key_val_str = key_val_dict_to_str(_format_metadata(self.metadata))
        string = '\n'.join([
            '1',
            f'{self.key} pbc="F F F"  Properties=species:S:1:pos:R:3 {key_val_str}',
            'X        0.00000000       0.00000000       0.00000000', ''
        ])
        return string


GeometryLike = Geometry | NullState
NULLSTATE = NullState()


def is_lower_triangular(cell: np.ndarray) -> bool:
    """
    Check if a cell matrix is lower triangular.

    Args:
        cell (np.ndarray): 3x3 cell matrix.

    Returns:
        bool: True if the cell matrix is lower triangular, False otherwise.
    """
    return (
        cell[0, 0] > 0
        and cell[1, 1] > 0  # positive volumes
        and cell[2, 2] > 0
        and cell[0, 1] == 0
        and cell[0, 2] == 0  # lower triangular
        and cell[1, 2] == 0
    )


def is_reduced(cell: np.ndarray) -> bool:
    """
    Check if a cell matrix is in reduced form.

    Args:
        cell (np.ndarray): 3x3 cell matrix.

    Returns:
        bool: True if the cell matrix is in reduced form, False otherwise.
    """
    return (
        cell[0, 0] > abs(2 * cell[1, 0])
        and cell[0, 0] > abs(2 * cell[2, 0])  # b mostly along y axis
        and cell[1, 1] > abs(2 * cell[2, 1])  # c mostly along z axis
        and is_lower_triangular(cell)  # c mostly along z axis
    )


def transform_lower_triangular(
    pos: np.ndarray, cell: np.ndarray, reorder: bool = False
):
    """Transforms coordinate axes such that cell matrix is lower diagonal

    The transformation is derived from the QR decomposition and performed
    in-place. Because the lower triangular form puts restrictions on the size
    of off-diagonal elements, lattice vectors are by default reordered from
    largest to smallest; this feature can be disabled using the reorder
    keyword.
    The box vector lengths and angles remain exactly the same.

    Args:
        pos (np.ndarray): Array of atomic positions.
        cell (np.ndarray): 3x3 cell matrix.
        reorder (bool, optional): Whether to reorder lattice vectors. Defaults to False.
    """
    if reorder:  # reorder box vectors as k, l, m with |k| >= |l| >= |m|
        norms = np.linalg.norm(cell, axis=1)
        ordering = np.argsort(norms)[::-1]  # largest first
        a = cell[ordering[0], :].copy()
        b = cell[ordering[1], :].copy()
        c = cell[ordering[2], :].copy()
        cell[0, :] = a[:]
        cell[1, :] = b[:]
        cell[2, :] = c[:]
    q, r = np.linalg.qr(cell.T)
    flip_vectors = np.eye(3) * np.diag(np.sign(r))  # reflections after rotation
    rotation = np.linalg.inv(q.T) @ flip_vectors  # full (improper) rotation
    pos[:] = pos @ rotation
    cell[:] = cell @ rotation
    assert np.allclose(cell, np.linalg.cholesky(cell @ cell.T), atol=1e-5)
    cell[0, 1] = 0
    cell[0, 2] = 0
    cell[1, 2] = 0


def reduce_box_vectors(cell: np.ndarray):
    """Uses linear combinations of box vectors to obtain the reduced form

    The reduced form of a cell matrix is lower triangular, with additional
    constraints that enforce vector b to lie mostly along the y-axis and vector
    c to lie mostly along the z axis.

    """
    # simple reduction algorithm only works on lower triangular cell matrices
    assert is_lower_triangular(cell)
    # replace c and b with shortest possible vectors to ensure
    # b_y > |2 c_y|
    # b_x > |2 c_x|
    # a_x > |2 b_x|
    cell[2, :] = cell[2, :] - cell[1, :] * np.round(cell[2, 1] / cell[1, 1])
    cell[2, :] = cell[2, :] - cell[0, :] * np.round(cell[2, 0] / cell[0, 0])
    cell[1, :] = cell[1, :] - cell[0, :] * np.round(cell[1, 0] / cell[0, 0])


@typeguard.typechecked
def get_mass_matrix(geometry: Geometry) -> np.ndarray:
    """
    Compute the mass matrix for a given geometry.

    Args:
        geometry (Geometry): Input geometry.

    Returns:
        np.ndarray: Mass matrix.
    """
    masses = np.repeat(
        np.array([atomic_masses[n] for n in geometry.numbers]),
        3,
    )
    sqrt_inv = 1 / np.sqrt(masses)
    return np.outer(sqrt_inv, sqrt_inv)


@typeguard.typechecked
def mass_weight(hessian: np.ndarray, geometry: Geometry) -> np.ndarray:
    """
    Apply mass-weighting to a Hessian matrix.

    Args:
        hessian (np.ndarray): Input Hessian matrix.
        geometry (Geometry): Geometry associated with the Hessian.

    Returns:
        np.ndarray: Mass-weighted Hessian matrix.
    """
    assert hessian.shape[0] == hessian.shape[1]
    assert len(geometry) * 3 == hessian.shape[0]
    return hessian * get_mass_matrix(geometry)


@typeguard.typechecked
def mass_unweight(hessian: np.ndarray, geometry: Geometry) -> np.ndarray:
    """
    Remove mass-weighting from a Hessian matrix.

    Args:
        hessian (np.ndarray): Input mass-weighted Hessian matrix.
        geometry (Geometry): Geometry associated with the Hessian.

    Returns:
        np.ndarray: Unweighted Hessian matrix.
    """
    assert hessian.shape[0] == hessian.shape[1]
    assert len(geometry) * 3 == hessian.shape[0]
    return hessian / get_mass_matrix(geometry)


@typeguard.typechecked
def get_unique_numbers(states: Sequence[Geometry]) -> set[int]:
    """Returns a set of unique atom numbers found across all states."""
    return set().union(*[np.unique(geom.numbers).tolist() for geom in states])


@typeguard.typechecked
def get_atomic_energy(geometry: Geometry, atomic_energies: dict[str, float]) -> float:
    """Compute the total atomic energy based on provided single atom energies."""
    total = 0
    numbers, counts = np.unique(geometry.numbers, return_counts=True)
    for number, count in zip(numbers, counts):
        symbol = chemical_symbols[number]
        try:
            total += count * atomic_energies[symbol]
        except KeyError:
            warnings.warn(f'No atomic energy value for symbol "{symbol}". Are you sure?')
    return float(total)


@typeguard.typechecked
def create_outputs(quantity_names: Sequence[str],
                   data: Sequence[GeometryLike]) -> tuple[dict[str, np.ndarray], set[str]]:
    """
    Create output arrays for specified quantities from a list of Geometry instances.

    Args:
        quantity_names (Sequence[str]): Sequence of quantity names to extract.
        data (Sequence[Geometry]): Sequence of Geometry instances.

    Returns:
        tuple[dict[str, np.ndarray], set[str]]: Dict of arrays with default values for every registered quantity
        and the set of unknown quantities
    """
    known_quantities = {k for k in quantity_names if k in quantities.all}
    natoms = [len(g) for g in data if g != NULLSTATE]
    max_natoms, nframes = np.max(natoms), len(data)

    array_dict = {}
    for name in known_quantities:
        q = quantities.all[name]
        shape = (max_natoms, *q.shape) if q.per_atom else q.shape
        if np.issubdtype(q.dtype, np.character):
            array = np.empty((nframes, *shape), dtype=object)       # string arrays require fixed length
        else:
            array = np.empty((nframes, *shape), dtype=q.dtype)
        array[:] = q.default
        array_dict[name] = array
    return array_dict, {k for k in quantity_names if k not in quantities.all}


def _assign_identifier(
    state: GeometryLike,
    identifier: int,
    discard: bool = False,
) -> tuple[Geometry, int]:
    """
    Assign an identifier to a Geometry instance.

    Args:
        state (Geometry): Input Geometry instance.
        identifier (int): Identifier to assign.
        discard (bool, optional): Whether to discard the state. Defaults to False.

    Returns:
        tuple[Geometry, int]: Updated Geometry and next available identifier.
    """
    if (state == NULLSTATE) or discard or 'identifier' in state.metadata:
        return state, identifier
    state['identifier'] = identifier
    return state, identifier + 1


assign_identifier = python_app(_assign_identifier, executors=["default_threads"])


@typeguard.typechecked
def _check_equality(
    state0: GeometryLike,
    state1: GeometryLike,
) -> bool:
    """
    Check if two Geometry instances are equal.

    Args:
        state0 (Geometry): First Geometry instance.
        state1 (Geometry): Second Geometry instance.

    Returns:
        bool: True if the Geometry instances are equal, False otherwise.
    """
    return state0 == state1


check_equality = python_app(_check_equality, executors=["default_threads"])



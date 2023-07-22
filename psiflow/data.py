"""
The `data` module implements objects used in the representation and IO of
atomic data.
An atomic configuration is defined by the number and cartesian coordinate of
each of its atoms as well as three noncoplanar box vectors which define the periodicity
of the system.
In addition, an atomic configuration may be *labeled* with the total potential
energy of the configuration, the atomic forces, and the virial stress tensor.
Finally, it can also contain pointers to
the output and error logs of QM evaluation calculations.

"""

from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Union, List
import typeguard
import os
import tempfile
import logging
import numpy as np
from pathlib import Path
from copy import deepcopy

from ase.data import chemical_symbols
from ase import Atoms

from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture
from parsl.dataflow.memoization import id_for_memo

import psiflow
from psiflow.utils import copy_data_future, copy_app_future, \
        resolve_and_check, transform_lower_triangular, \
        reduce_box_vectors, is_reduced


logger = logging.getLogger(__name__) # logging per module


@typeguard.typechecked
class FlowAtoms(Atoms):
    """Wrapper class around ASE `Atoms` with additional attributes for QM logs

    In addition to the standard `Atoms` functionality, this class offers the
    ability to store pointers to output and error logs that have been generated
    during a QM evaluation of the atomic structure. A separate attribute
    is reserved to store the exit code of the calculation (success or failed)
    as a boolean.

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if 'reference_stdout' not in self.info.keys(): # only set if not present
            self.info['reference_stdout'] = False # default None not supported
        if 'reference_stderr' not in self.info.keys(): # only set if not present
            self.info['reference_stderr'] = False
        if 'reference_status' not in self.info.keys(): # only set if not present
            self.info['reference_status'] = False

    @property
    def reference_status(self) -> bool:
        """True if QM evaluation was successful, False otherwise"""
        return self.info['reference_status']

    @reference_status.setter
    def reference_status(self, flag: bool) -> None:
        assert flag in [True, False]
        self.info['reference_status'] = flag

    @property
    def reference_stdout(self) -> Union[bool, str]:
        """Contains filepath to QM output log, False if not yet performed"""
        return self.info['reference_stdout']

    @reference_stdout.setter
    def reference_stdout(self, path: Union[bool, str]) -> None:
        self.info['reference_stdout'] = path

    @property
    def reference_stderr(self) -> Union[bool, str]:
        """Contains filepath to QM error log, False if not yet performed"""
        return self.info['reference_stderr']

    @reference_stderr.setter
    def reference_stderr(self, path: Union[bool, str]) -> None:
        self.info['reference_stderr'] = path

    @property
    def elements(self) -> list[str]:
        numbers = set([n for n in self.numbers])
        return [chemical_symbols[n] for n in numbers]

    def reset(self) -> None:
        info = {}
        for key, value in self.info.items():
            if 'energy' in key: # atomic, formation, or total energy
                pass
            elif 'stress' in key: # stress
                pass
            else:
                info[key] = value
        info['reference_stdout'] = False
        info['reference_stderr'] = False
        info['reference_status'] = False
        self.calc = None # necessary
        self.info = info
        self.arrays.pop('forces', None)

    def canonical_orientation(self):
        pos = self.get_positions()
        box = np.array(self.get_cell())
        transform_lower_triangular(pos, box, reorder=False)
        reduce_box_vectors(box)
        self.set_positions(pos)
        self.set_cell(box)

    @classmethod
    def from_atoms(cls, atoms: Atoms) -> FlowAtoms:
        """Generates a `FlowAtoms` object based on an existing `Atoms`

        Array attributes need to be copied manually as this is for some reason
        not done by the ASE constructor.

        Args:
            atoms (Atoms):
                contains atomic configuration to be stored as `FlowAtoms`

        """
        from copy import deepcopy
        flow_atoms = deepcopy(atoms)
        flow_atoms.__class__ = FlowAtoms
        if 'reference_stdout' not in flow_atoms.info.keys(): # only set if not present
            flow_atoms.info['reference_stdout'] = False # default None not supported
        if 'reference_stderr' not in flow_atoms.info.keys(): # only set if not present
            flow_atoms.info['reference_stderr'] = False
        if 'reference_status' not in flow_atoms.info.keys(): # only set if not present
            flow_atoms.info['reference_status'] = False
        return flow_atoms


# use universal dummy state
NullState = FlowAtoms(numbers=[0], positions=[[0, 0, 0]])


@typeguard.typechecked
def _canonical_orientation(
        inputs: list[File] = [],
        outputs: list[File] = [],
        ) -> None:
    from psiflow.data import read_dataset
    from ase.io.extxyz import write_extxyz
    data = read_dataset(slice(None), inputs=[inputs[0]])
    for atoms in data:
        atoms.canonical_orientation()
    with open(outputs[0], 'w') as f:
        write_extxyz(f, data)
canonical_orientation = python_app(_canonical_orientation, executors=['default'])


@typeguard.typechecked
def reset_atoms(atoms: Union[Atoms, FlowAtoms]): # modify FlowAtoms Future before returning
    from copy import deepcopy
    _atoms = deepcopy(atoms)
    if not type(_atoms) == FlowAtoms:
        _atoms = FlowAtoms.from_atoms(_atoms)
    _atoms.reset()
    return _atoms
app_reset_atoms = python_app(reset_atoms, executors=['default'])


@typeguard.typechecked
def write_dataset(
        states: Optional[List[Optional[FlowAtoms]]],
        inputs: List[Optional[FlowAtoms]] = [], # allow None
        return_data: bool = False, # whether to return data
        outputs: List[File] = [],
        ) -> Optional[List[FlowAtoms]]:
    from ase.io.extxyz import write_extxyz
    if states is not None:
        _data = states
    else:
        _data = inputs
    with open(outputs[0], 'w') as f:
        write_extxyz(f, _data)
    if return_data:
        return _data
app_write_dataset = python_app(write_dataset, executors=['default'])


@typeguard.typechecked
def _write_atoms(atoms: FlowAtoms, outputs=[]):
    from ase.io import write
    write(outputs[0].filepath, atoms)
write_atoms = python_app(_write_atoms, executors=['default'])


@typeguard.typechecked
def read_dataset(
        index_or_indices: Union[int, List[int], slice],
        inputs: List[File] = [],
        outputs: List[File] = [],
        ) -> Union[FlowAtoms, List[FlowAtoms]]:
    from ase.io.extxyz import read_extxyz, write_extxyz
    from psiflow.data import FlowAtoms
    with open(inputs[0], 'r' ) as f:
        if type(index_or_indices) == int:
            atoms = list(read_extxyz(f, index=index_or_indices))[0]
            data  = FlowAtoms.from_atoms(atoms) # single atoms instance
        else:
            if type(index_or_indices) == list:
                data = [list(read_extxyz(f, index=i))[0] for i in index_or_indices]
            elif type(index_or_indices) == slice:
                data = list(read_extxyz(f, index=index_or_indices))
            else:
                raise ValueError
            data = [FlowAtoms.from_atoms(a) for a in data] # list of atoms
    if len(outputs) > 0: # write to file
        with open(outputs[0], 'w') as f:
            write_extxyz(f, data)
    return data
app_read_dataset = python_app(read_dataset, executors=['default'])


@typeguard.typechecked
def reset_dataset(
        inputs: List[File] = [],
        outputs: List[File] = [],
        ) -> None:
    from psiflow.data import read_dataset
    from ase.io.extxyz import write_extxyz
    data = read_dataset(slice(None), inputs=[inputs[0]])
    for atoms in data:
        atoms.reset()
    with open(outputs[0], 'w') as f:
        write_extxyz(f, data)
app_reset_dataset = python_app(reset_dataset, executors=['default'])


@typeguard.typechecked
def join_dataset(inputs: List[File] = [], outputs: List[File] = []) -> None:
    data = []
    for i in range(len(inputs)):
        data += read_dataset(slice(None), inputs=[inputs[i]]) # read all
    write_dataset(data, outputs=[outputs[0]])
app_join_dataset = python_app(join_dataset, executors=['default'])


@typeguard.typechecked
def get_length_dataset(inputs: List[File] = []) -> int:
    data = read_dataset(slice(None), inputs=[inputs[0]])
    return len(data)
app_length_dataset = python_app(get_length_dataset, executors=['default'])


@typeguard.typechecked
def _get_indices(
        flag: str,
        inputs: List[File] = [],
        ) -> List[int]:
    from psiflow.data import read_dataset, NullState
    data = read_dataset(slice(None), inputs=[inputs[0]])
    indices = []
    for i, atoms in enumerate(data):
        if flag == 'labeled':
            if atoms.reference_status:
                indices.append(i)
        elif flag == 'not_null':
            if atoms != NullState:
                indices.append(i)
        else:
            raise ValueError('unrecopgnized flag ' + flag)
    return indices
get_indices = python_app(_get_indices, executors=['default'])


@typeguard.typechecked
def compute_metrics(
        intrinsic: bool,
        atom_indices: Optional[List[int]],
        elements: Optional[List[str]],
        metric: str,
        properties: List[str],
        inputs: List[File] = [],
        ) -> np.ndarray:
    import numpy as np
    from copy import deepcopy
    from ase.units import Pascal
    from psiflow.data import read_dataset
    from psiflow.utils import get_index_element_mask, compute_error
    data_0 = read_dataset(slice(None), inputs=[inputs[0]])
    if len(inputs) == 1:
        assert intrinsic
        data_1 = [deepcopy(a) for a in data_0]
        for atoms_1 in data_1:
            if 'energy' in atoms_1.info.keys():
                atoms_1.info['energy'] = 0.0
            if 'formation_energy' in atoms_1.info.keys():
                atoms_1.info['formation_energy'] = 0.0
            if 'stress' in atoms_1.info.keys(): # ASE copy fails for info attrs!
                atoms_1.info['stress'] = np.zeros((3, 3))
            if 'forces' in atoms_1.arrays.keys():
                atoms_1.arrays['forces'][:] = 0.0
    else:
        data_1 = read_dataset(slice(None), inputs=[inputs[1]])
    assert len(data_0) == len(data_1)
    for atoms_0, atoms_1 in zip(data_0, data_1):
        assert np.allclose(atoms_0.numbers, atoms_1.numbers)
        assert np.allclose(atoms_0.positions, atoms_1.positions)
        if atoms_0.cell is not None:
            assert np.allclose(atoms_0.cell, atoms_1.cell)

    errors = np.zeros((len(data_0), len(properties)))
    outer_mask = np.array([True] * len(data_0))
    for i in range(len(data_0)):
        atoms_0 = data_0[i]
        atoms_1 = data_1[i]
        if (atom_indices is not None) or (elements is not None):
            assert 'energy' not in properties
            assert 'stress' not in properties
            assert 'forces' in properties # only makes sense for forces
            mask = get_index_element_mask(atoms_0.numbers, elements, atom_indices)
        else:
            mask = np.array([True] * len(atoms_0))
        if not np.any(mask): # no target atoms present; skip
            outer_mask[i] = False
            continue
    if not np.any(outer_mask):
        raise AssertionError('no states in dataset contained atoms of interest')
    for i in range(len(data_0)):
        atoms_0 = data_0[i]
        atoms_1 = data_1[i]
        if outer_mask[i]:
            errors[i, :] = compute_error(
                    atoms_0,
                    atoms_1,
                    atom_indices,
                    elements,
                    metric,
                    properties,
                    )
    return errors[outer_mask, :]
app_compute_metrics = python_app(compute_metrics, executors=['default'])


@typeguard.typechecked
def insert_formation_energy(
        elements: list[str],
        inputs: list[Union[File, float]] = [],
        outputs: list[File] = [],
        ) -> None:
    import numpy as np
    from ase.data import atomic_numbers
    from ase.io.extxyz import write_extxyz
    assert len(inputs) == len(elements) + 1
    data = read_dataset(slice(None), inputs=[inputs[0]])
    numbers = [atomic_numbers[e] for e in elements]
    for atoms in data:
        reference = 0
        indices = []
        natoms = len(atoms)
        for i, number in enumerate(numbers):
            natoms_per_number = np.sum(atoms.numbers == number)
            if natoms_per_number == 0:
                continue
            element = elements[i]
            energy = inputs[1 + i]
            label = 'atomic_energy_{}'.format(element)
            if label in atoms.info.keys():
                assert np.allclose(energy, atoms.info[label]), (
                        'attempting to insert atomic energy {} for '
                        'element {} into dataset but another value '
                        'was already present: {}'.format(
                            energy,
                            element,
                            atoms.info[label],
                            ))
            else:
                atoms.info[label] = energy
            reference += natoms_per_number * energy
            natoms -= natoms_per_number
        assert natoms == 0 # all atoms accounted for
        atoms.info['formation_energy'] = atoms.info['energy'] - reference
    with open(outputs[0], 'w') as f:
        write_extxyz(f, data)
app_insert_formation_energy = python_app(insert_formation_energy, executors=['default'])


@typeguard.typechecked
def get_elements(inputs=[]) -> set[str]:
    from ase.data import chemical_symbols
    data = read_dataset(slice(None), inputs=[inputs[0]])
    return set([e for atoms in data for e in atoms.elements])
app_get_elements = python_app(get_elements, executors=['default'])


@typeguard.typechecked
def get_info_keys(inputs=[]) -> list[str]:
    data = read_dataset(slice(None), inputs=[inputs[0]])
    labels = list(data[0].info.keys())
    for atoms in data:
        for label in list(labels):
            if not label in atoms.info:
                labels.remove(label)
    return labels
app_get_info_keys = python_app(get_info_keys, executors=['default'])


@typeguard.typechecked
def get_energy_labels(info_keys) -> list[str]:
    labels = []
    for key in info_keys:
        if 'energy' in key:
            labels.append(key)
    return labels
app_get_energy_labels = python_app(get_energy_labels, executors=['default'])


@typeguard.typechecked
class Dataset:
    """Container to represent a dataset of atomic structures

    Args:
        context: an `ExecutionContext` instance with a 'default' executor.
        atoms_list: a list of `Atoms` instances which represent the dataset.
        data_future: a `parsl.app.futures.DataFuture` instance that points
            to an `.xyz` file.

    """

    def __init__(
            self,
            atoms_list: Optional[Union[List[AppFuture], List[Union[FlowAtoms, Atoms]], AppFuture]],
            data_future: Optional[Union[DataFuture, File]] = None,
            ) -> None:
        context = psiflow.context()

        if data_future is None: # generate new DataFuture
            assert atoms_list is not None
            if isinstance(atoms_list, AppFuture):
                states = atoms_list
                inputs = []
            else:
                if (len(atoms_list) > 0) and isinstance(atoms_list[0], AppFuture):
                    states = None
                    inputs = atoms_list
                else:
                    states = [FlowAtoms.from_atoms(a) for a in atoms_list]
                    inputs = []
            self.data_future = app_write_dataset(
                    states,
                    inputs=inputs,
                    outputs=[context.new_file('data_', '.xyz')],
                    ).outputs[0]
        else:
            assert atoms_list is None # do not allow additional atoms
            self.data_future = copy_data_future(
                    inputs=[data_future],
                    outputs=[context.new_file('data_', '.xyz')],
                    ).outputs[0] # ensure type(data_future) == DataFuture

    def info_keys(self) -> AppFuture:
        return app_get_info_keys(inputs=[self.data_future])

    def energy_labels(self) -> AppFuture:
        return app_get_energy_labels(self.info_keys())

    def length(self) -> AppFuture:
        return app_length_dataset(inputs=[self.data_future])

    def shuffle(self):
        indices = np.arange(self.length().result())
        np.random.shuffle(indices)
        return self.get(indices=[int(i) for i in indices])

    def __getitem__(
            self,
            index: Union[int, slice, List[int], AppFuture],
            ) -> Union[Dataset, AppFuture]:
        if isinstance(index, int):
            return self.get(index=index)
        else: # slice, List, AppFuture
            return self.get(indices=index)

    def get(
            self,
            index: Optional[int] = None,
            indices: Optional[Union[List[int], AppFuture, slice]] = None,
            ) -> Union[Dataset, AppFuture]:
        context = psiflow.context()
        if indices is not None:
            assert index is None
            data_future = app_read_dataset(
                    indices,
                    inputs=[self.data_future],
                    outputs=[context.new_file('data_', '.xyz')],
                    ).outputs[0]
            return Dataset(None, data_future=data_future)
        else:
            assert index is not None
            atoms = app_read_dataset(
                    index, # int or AppFuture of int
                    inputs=[self.data_future],
                    ) # represents an AppFuture of an ase.Atoms instance
            return atoms

    def save(
            self,
            path_dataset: Union[Path, str],
            require_done: bool = True,
            ) -> AppFuture:
        path_dataset = resolve_and_check(Path(path_dataset))
        future = copy_data_future(
                inputs=[self.data_future],
                outputs=[File(str(path_dataset))],
                )
        if require_done:
            future.result()
        return future

    def as_list(self) -> AppFuture:
        return app_read_dataset(
                index_or_indices=slice(None),
                inputs=[self.data_future],
                )

    def append(self, dataset: Dataset) -> None:
        context = psiflow.context()
        self.data_future = app_join_dataset(
                inputs=[self.data_future, dataset.data_future],
                outputs=[context.new_file('data_', '.xyz')],
                ).outputs[0]

    def __add__(self, dataset: Dataset) -> Dataset:
        context = psiflow.context()
        data_future = app_join_dataset(
                inputs=[self.data_future, dataset.data_future],
                outputs=[context.new_file('data_', '.xyz')],
                ).outputs[0]
        return Dataset(None, data_future)

    def log(self, name):
        logger.info('dataset {} contains {} states'.format(name, self.length().result()))

    def set_formation_energy(self, **atomic_energies) -> Dataset:
        context = psiflow.context()
        elements = list(atomic_energies.keys())
        energies = [atomic_energies[e] for e in elements]
        data_future = app_insert_formation_energy(
                elements,
                inputs=[self.data_future] + energies,
                outputs=[context.new_file('data_', '.xyz')],
                ).outputs[0]
        return Dataset(None, data_future)

    def elements(self):
        return app_get_elements(inputs=[self.data_future])

    def reset(self):
        context = psiflow.context()
        data_future = app_reset_dataset(
                inputs=[self.data_future],
                outputs=[context.new_file('data_', '.xyz')],
                ).outputs[0]
        return Dataset(None, data_future)

    def labeled(self) -> Dataset:
        indices = get_indices(
                'labeled',
                inputs=[self.data_future],
                )
        return self.get(indices=indices)

    def not_null(self) -> Dataset:
        indices = get_indices(
                'not_null',
                inputs=[self.data_future],
                )
        return self.get(indices=indices)

    def canonical_orientation(self):
        future = canonical_orientation(
                inputs=[self.data_future],
                outputs=[psiflow.context().new_file('data_', '.xyz')],
                )
        return Dataset(None, data_future=future.outputs[0])

    @staticmethod
    def get_errors(
            dataset_0: Dataset,
            dataset_1: Optional[Dataset], # None when computing intrinsic errors
            atom_indices: Optional[List[int]] = None,
            elements: Optional[List[str]] = None,
            metric: str = 'rmse',
            properties: List[str] = ['energy', 'forces', 'stress'],
            ) -> AppFuture:
        inputs = [dataset_0.data_future]
        if dataset_1 is not None:
            inputs.append(dataset_1.data_future)
            intrinsic = False
        else:
            intrinsic = True
        return app_compute_metrics(
                intrinsic=intrinsic,
                atom_indices=atom_indices,
                elements=elements,
                metric=metric,
                properties=properties,
                inputs=inputs,
                )

    @classmethod
    def load(
            cls,
            path_xyz: Union[Path, str],
            ) -> Dataset:
        path_xyz = resolve_and_check(Path(path_xyz))
        assert os.path.isfile(path_xyz) # needs to be locally accessible
        context = psiflow.context()
        return cls(None, data_future=File(str(path_xyz)))

    @staticmethod
    def create_apps() -> None:
        pass # no apps beyond default executor

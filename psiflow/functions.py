import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional, Type, Union, get_type_hints

import numpy as np
import typeguard
from ase import Atoms
from ase.data import atomic_masses, chemical_symbols
from ase.units import fs, kJ, mol, nm

from psiflow.geometry import Geometry, NullState, create_outputs


@typeguard.typechecked
@dataclass
class Function:
    outputs: ClassVar[tuple]

    def __call__(
        self,
        geometries: list[Geometry],
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError


@dataclass
class EnergyFunction(Function):
    outputs: ClassVar[tuple[str, ...]] = ("energy", "forces", "stress")


@typeguard.typechecked
@dataclass
class EinsteinCrystalFunction(EnergyFunction):
    force_constant: float
    centers: np.ndarray
    volume: float = 0.0

    def __call__(
        self,
        geometries: list[Geometry],
    ) -> dict[str, np.ndarray]:
        value, grad_pos, grad_cell = create_outputs(
            self.outputs,
            geometries,
        )

        for i, geometry in enumerate(geometries):
            if geometry == NullState:
                continue

            delta = geometry.per_atom.positions - self.centers
            value[i] = self.force_constant * np.sum(delta**2) / 2
            grad_pos[i, : len(geometry)] = (-1.0) * self.force_constant * delta
            if geometry.periodic and self.volume > 0.0:
                delta = np.linalg.det(geometry.cell) - self.volume
                _stress = self.force_constant * np.eye(3) * delta
            else:
                _stress = np.zeros((3, 3))
            grad_cell[i, :] = _stress

        return {"energy": value, "forces": grad_pos, "stress": grad_cell}


@typeguard.typechecked
@dataclass
class PlumedFunction(EnergyFunction):
    plumed_input: str
    external: Optional[Union[str, Path]] = None

    def __post_init__(self):
        self.plumed_instances = {}

    def __call__(
        self,
        geometries: list[Geometry],
    ) -> dict[str, np.ndarray]:
        value, grad_pos, grad_cell = create_outputs(
            self.outputs,
            geometries,
        )

        plumed_input = self.plumed_input
        if self.external is not None:
            assert self.external in plumed_input

        def geometry_to_key(geometry: Geometry) -> tuple:
            return tuple([geometry.periodic]) + tuple(geometry.per_atom.numbers)

        for i, geometry in enumerate(geometries):
            if geometry == NullState:
                continue
            key = geometry_to_key(geometry)
            if key not in self.plumed_instances:
                from plumed import Plumed

                tmp = tempfile.NamedTemporaryFile(
                    prefix="plumed_", mode="w+", delete=False
                )
                # plumed creates a back up if this file would already exist
                os.remove(tmp.name)
                plumed_ = Plumed()
                ps = 1000 * fs
                plumed_.cmd("setRealPrecision", 8)
                plumed_.cmd("setMDEnergyUnits", mol / kJ)
                plumed_.cmd("setMDLengthUnits", 1 / nm)
                plumed_.cmd("setMDTimeUnits", 1 / ps)
                plumed_.cmd("setMDChargeUnits", 1.0)
                plumed_.cmd("setMDMassUnits", 1.0)

                plumed_.cmd("setLogFile", tmp.name)
                plumed_.cmd("setRestart", True)
                plumed_.cmd("setNatoms", len(geometry))
                plumed_.cmd("init")
                plumed_input = self.plumed_input
                for line in plumed_input.split("\n"):
                    plumed_.cmd("readInputLine", line)
                os.remove(tmp.name)  # remove whatever plumed has created
                self.plumed_instances[key] = plumed_

            plumed_ = self.plumed_instances[key]
            if geometry.periodic:
                cell = np.copy(geometry.cell).astype(np.float64)
                plumed_.cmd("setBox", cell)

            # set positions
            energy = np.zeros((1,))
            forces = np.zeros((len(geometry), 3))
            virial = np.zeros((3, 3))
            masses = np.array([atomic_masses[n] for n in geometry.per_atom.numbers])
            plumed_.cmd("setStep", 0)
            plumed_.cmd("setPositions", geometry.per_atom.positions.astype(np.float64))
            plumed_.cmd("setMasses", masses)
            plumed_.cmd("setForces", forces)
            plumed_.cmd("setVirial", virial)
            plumed_.cmd("prepareCalc")
            plumed_.cmd("performCalcNoUpdate")
            plumed_.cmd("getBias", energy)
            if geometry.periodic:
                stress = virial / np.linalg.det(geometry.cell)
            else:
                stress = np.zeros((3, 3))

            value[i] = float(energy.item())
            grad_pos[i, : len(geometry)] = forces
            grad_cell[i] = stress
        return {"energy": value, "forces": grad_pos, "stress": grad_cell}


@typeguard.typechecked
class ZeroFunction(EnergyFunction):

    def __call__(
        self,
        geometries: list[Geometry],
    ) -> dict[str, np.ndarray]:
        energy, forces, stress = create_outputs(
            self.outputs,
            geometries,
        )
        for i, geometry in enumerate(geometries):
            if geometry == NullState:
                continue
            energy[i] = 0.0
            forces[i, : len(geometry)] = 0.0
            stress[i, :] = 0.0
        return {"energy": energy, "forces": forces, "stress": stress}


@typeguard.typechecked
@dataclass
class HarmonicFunction(EnergyFunction):
    positions: np.ndarray
    hessian: np.ndarray
    energy: Optional[float] = None

    def __call__(self, geometries: list[Geometry]) -> dict[str, np.ndarray]:
        self.positions = np.array(self.positions)
        self.hessian = np.array(self.hessian)

        energy, forces, stress = create_outputs(self.outputs, geometries)

        for i, geometry in enumerate(geometries):
            if geometry == NullState:
                continue
            delta = geometry.per_atom.positions.reshape(-1) - self.positions.reshape(-1)
            grad = np.dot(self.hessian, delta)
            energy[i] = 0.5 * np.dot(delta, grad)
            if self.energy is not None:
                energy[i] += self.energy
            forces[i] = (-1.0) * grad.reshape(-1, 3)
            stress[i] = 0.0
        return {"energy": energy, "forces": forces, "stress": stress}


@typeguard.typechecked
@dataclass
class MACEFunction(EnergyFunction):
    model_path: str
    ncores: int
    device: str
    dtype: str
    atomic_energies: dict[str, float]
    env_vars: Optional[dict[str, str]] = None

    def __post_init__(self):
        import logging
        import os

        # import environment variables before any nontrivial imports
        if self.env_vars is not None:
            for key, value in self.env_vars.items():
                os.environ[key] = value

        import torch
        from mace.tools import torch_tools, utils

        torch_tools.set_default_dtype(self.dtype)
        if self.device == 'gpu':  # when it's not a specific GPU, use any
            self.device = 'cuda'
        self.device = torch_tools.init_device(self.device)

        torch.set_num_threads(self.ncores)
        model = torch.load(f=self.model_path, map_location="cpu")
        if self.dtype == "float64":
            model = model.double()
        else:
            model = model.float()
        model = model.to(self.device)
        model.eval()
        self.model = model
        self.r_max = float(self.model.r_max)
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )

        # remove unwanted streamhandler added by MACE / torch!
        logging.getLogger("").removeHandler(logging.getLogger("").handlers[0])

    def get_atomic_energy(self, geometry):
        total = 0
        natoms = len(geometry)
        for number in list(set(geometry.per_atom.numbers)):
            natoms_per_symbol = np.sum(geometry.per_atom.numbers == number)
            if natoms_per_symbol == 0:
                continue
            symbol = chemical_symbols[number]
            total += natoms_per_symbol * self.atomic_energies[symbol]
            natoms -= natoms_per_symbol
        assert natoms == 0  # all atoms accounted for
        return total

    def __call__(self, geometries: list[Geometry]) -> dict[str, np.ndarray]:
        from mace import data
        from mace.tools import torch_geometric, torch_tools

        torch_tools.set_default_dtype(self.dtype)

        energy, forces, stress = create_outputs(self.outputs, geometries)

        for i, geometry in enumerate(geometries):
            if geometry == NullState:
                continue

            # compute offset if possible
            if len(self.atomic_energies) > 0:
                energy[i] = self.get_atomic_energy(geometry)
            else:
                energy[i] = 0.0

            cell = None
            if geometry.periodic:
                cell = np.copy(geometry.cell)
            atoms = Atoms(
                positions=geometry.per_atom.positions,
                numbers=geometry.per_atom.numbers,
                cell=cell,
                pbc=geometry.periodic,
            )
            config = data.config_from_atoms(atoms)
            data_loader = torch_geometric.dataloader.DataLoader(
                dataset=[
                    data.AtomicData.from_config(
                        config, z_table=self.z_table, cutoff=self.r_max
                    )
                ],
                batch_size=1,
                shuffle=False,
                drop_last=False,
            )
            batch = next(iter(data_loader)).to(device=self.device)

            compute_stress = cell is not None
            out = self.model(batch.to_dict(), compute_stress=compute_stress)

            energy[i] += out["energy"].detach().cpu().item()
            forces[i, : len(geometry)] = out["forces"].detach().cpu().numpy()
            if cell is not None:
                stress[i, :] = out["stress"].detach().cpu().numpy()
        return {"energy": energy, "forces": forces, "stress": stress}


@staticmethod
def sort_outputs(
    outputs_: list[str],
    **kwargs,
) -> list[np.ndarray]:
    output_arrays = []
    for name in outputs_:
        array = kwargs.get(name, None)
        assert array is not None
        output_arrays.append(array)
    return output_arrays


def _apply(
    arg: Union[Geometry, list[Geometry], None],
    outputs_: tuple[str, ...],
    inputs: list = [],
    function_cls: Optional[Type[Function]] = None,
    parsl_resource_specification: dict = {},
    **parameters,
) -> Optional[list[np.ndarray]]:
    from psiflow.data.utils import _read_frames

    assert function_cls is not None
    if arg is None:
        states = _read_frames(inputs=[inputs[0]])
    elif not isinstance(arg, list):
        states = [arg]
    else:
        states = arg
    function = function_cls(**parameters)
    output_dict = function(states)
    output_arrays = sort_outputs(outputs_, **output_dict)
    return output_arrays


def function_from_json(path: Union[str, Path], **kwargs) -> Function:
    functions = [
        EinsteinCrystalFunction,
        HarmonicFunction,
        MACEFunction,
        PlumedFunction,
        None,
    ]
    with open(path, "r") as f:
        data = json.loads(f.read())
    assert "function_name" in data
    for function_cls in functions:
        if data["function_name"] == function_cls.__name__:
            break
    data.pop("function_name")
    for name, type_hint in get_type_hints(function_cls).items():
        if type_hint is np.ndarray:
            data[name] = np.array(data[name])
    for key, value in kwargs.items():
        if key in data:
            data[key] = value
    function = function_cls(**data)
    return function
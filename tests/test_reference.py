import requests
import pytest
import os
import molmod
import numpy as np
from pathlib import Path
from parsl.dataflow.futures import AppFuture
from parsl.app.futures import DataFuture

from pymatgen.io.cp2k.inputs import Cp2kInput

from ase import Atoms
from ase.io.extxyz import write_extxyz
from ase.units import Pascal

import psiflow
from psiflow.data import FlowAtoms, NullState
from psiflow.reference import EMTReference, CP2KReference, MP2CP2KReference
from psiflow.reference._cp2k import insert_filepaths_in_input, \
        insert_atoms_in_input
from psiflow.data import Dataset


@pytest.fixture
def fake_cp2k_input():
    return  """
&FORCE_EVAL
   METHOD Quickstep
   STRESS_TENSOR ANALYTICAL
   &DFT
      UKS  F
      MULTIPLICITY  1
      BASIS_SET_FILE_NAME  /user/gent/425/vsc42527/scratch/cp2k/SOURCEFILES/BASISSETS
      POTENTIAL_FILE_NAME  /user/gent/425/vsc42527/scratch/cp2k/SOURCEFILES/GTH_POTENTIALS
      &XC
         &VDW_POTENTIAL
            &PAIR_POTENTIAL
               PARAMETER_FILE_NAME  /user/gent/425/vsc42527/scratch/cp2k/SOURCEFILES/dftd3.dat
            &END PAIR_POTENTIAL
         &END VDW_POTENTIAL
      &END XC
   &END DFT
   &SUBSYS
      &KIND Al
         ELEMENT  H
         BASIS_SET foo
         POTENTIAL bar
      &END KIND
      &COORD
         H 4.0 0.0 0.0
      &END COORD
    &END SUBSYS
&END FORCE_EVAL
"""


@pytest.fixture
def cp2k_data():
    basis     = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/BASIS_MOLOPT_UZH').text
    dftd3     = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/dftd3.dat').text
    potential = requests.get('https://raw.githubusercontent.com/cp2k/cp2k/v9.1.0/data/POTENTIAL_UZH').text
    return {
            'basis_set': basis,
            'potential': potential,
            'dftd3': dftd3,
            }


@pytest.fixture
def cp2k_input():
    # remove stress tensor keyword to ensure it gets added automatically
    return """
&FORCE_EVAL
   METHOD Quickstep
   !STRESS_TENSOR ANALYTICAL
   &DFT
      UKS  F
      MULTIPLICITY  1
      BASIS_SET_FILE_NAME  dummy
      POTENTIAL_FILE_NAME  dummy
      &SCF
         MAX_SCF  10
         MAX_DIIS  8
         EPS_SCF  1.0E-06
         SCF_GUESS  RESTART
         &OT
            MINIMIZER  CG
            PRECONDITIONER  FULL_SINGLE_INVERSE
         &END OT
         &OUTER_SCF T
            MAX_SCF  10
            EPS_SCF  1.0E-06
         &END OUTER_SCF
      &END SCF
      &QS
         METHOD  GPW
         EPS_DEFAULT  1.0E-4
         EXTRAPOLATION  USE_GUESS
      &END QS
      &MGRID
         REL_CUTOFF [Ry]  60.0
         NGRIDS  5
         CUTOFF [Ry] 1000
      &END MGRID
      &XC
         DENSITY_CUTOFF   1.0E-10
         GRADIENT_CUTOFF  1.0E-10
         TAU_CUTOFF       1.0E-10
         &XC_FUNCTIONAL PBE
         &END XC_FUNCTIONAL
         &VDW_POTENTIAL
            POTENTIAL_TYPE  PAIR_POTENTIAL
            &PAIR_POTENTIAL
               TYPE  DFTD3(BJ)
               PARAMETER_FILE_NAME  parameter
               REFERENCE_FUNCTIONAL PBE
               R_CUTOFF  25
            &END PAIR_POTENTIAL
         &END VDW_POTENTIAL
      &END XC
   &END DFT
   &SUBSYS
      &KIND H
         ELEMENT  H
         BASIS_SET TZVP-MOLOPT-PBE-GTH-q1
         POTENTIAL GTH-PBE-q1
      &END KIND
      &KIND O
         ELEMENT  O
         BASIS_SET TZVP-MOLOPT-PBE-GTH-q6
         POTENTIAL GTH-PBE-q6
      &END KIND
      &KIND Si
         ELEMENT  Si
         BASIS_SET TZVP-MOLOPT-PBE-GTH-q4
         POTENTIAL GTH-PBE-q4
      &END KIND
      &KIND C
         ELEMENT  C
         BASIS_SET TZVP-MOLOPT-PBE-GTH-q4
         POTENTIAL GTH-PBE-q4
      &END KIND
      &KIND Al
         ELEMENT  Al
         BASIS_SET TZVP-MOLOPT-PBE-GTH-q3
         POTENTIAL GTH-PBE-q3
      &END KIND
   &END SUBSYS
!   &PRINT
!      &STRESS_TENSOR ON
!      &END STRESS_TENSOR
!      &FORCES
!      &END FORCES
!   &END PRINT
&END FORCE_EVAL
"""


@pytest.fixture
def cp2k_reference(context, cp2k_input, cp2k_data, tmp_path):
    reference = CP2KReference(cp2k_input=cp2k_input)
    for key, value in cp2k_data.items():
        with open(tmp_path / key, 'w') as f:
            f.write(value)
        reference.add_file(key, tmp_path / key)
    return reference


def test_reference_emt(context, dataset, tmp_path):
    reference = EMTReference()
    # modify dataset to include states for which EMT fails:
    _ = reference.evaluate(dataset).as_list().result()
    atoms_list = dataset.as_list().result()
    atoms_list[6].numbers[1] = 90
    atoms_list[9].numbers[1] = 3
    dataset_ = Dataset(atoms_list)
    evaluated = reference.evaluate(dataset_)
    assert evaluated.length().result() == len(atoms_list)

    atoms = reference.evaluate(dataset_[5]).result()
    assert type(atoms) == FlowAtoms
    assert atoms.reference_status == True
    atoms = reference.evaluate(dataset_[6]).result()
    assert type(atoms) == FlowAtoms
    assert atoms.reference_status == False


def test_cp2k_insert_filepaths(fake_cp2k_input):
    filepaths = {
            'basis_set': 'basisset0',
            'basis_giggle': 'basisset1',
            'potential': 'potential',
            'dftd3': 'parameter',
            }
    target_input = """
&FORCE_EVAL
   METHOD Quickstep
   STRESS_TENSOR ANALYTICAL
   &DFT
      UKS  F
      MULTIPLICITY  1
      POTENTIAL_FILE_NAME  potential
      BASIS_SET_FILE_NAME  basisset0
      BASIS_SET_FILE_NAME  basisset1
      &XC
         &VDW_POTENTIAL
            &PAIR_POTENTIAL
               PARAMETER_FILE_NAME  parameter
            &END PAIR_POTENTIAL
         &END VDW_POTENTIAL
      &END XC
   &END DFT
   &SUBSYS
      &KIND Al
         ELEMENT  H
         BASIS_SET foo
         POTENTIAL bar
      &END KIND
      &COORD
         H 4.0 0.0 0.0
      &END COORD
    &END SUBSYS
&END FORCE_EVAL
"""
    target = Cp2kInput.from_string(target_input)
    sample = Cp2kInput.from_string(insert_filepaths_in_input(fake_cp2k_input, filepaths))
    assert str(target) == str(sample)


def test_cp2k_success(context, cp2k_reference):
    atoms = FlowAtoms( # simple H2 at ~optimized interatomic distance
            numbers=np.ones(2),
            cell=5 * np.eye(3),
            positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
            pbc=True,
            )
    dataset = Dataset([atoms])
    evaluated = cp2k_reference.evaluate(dataset[0])
    assert isinstance(evaluated, AppFuture)
    assert evaluated.result().reference_status == True
    assert Path(evaluated.result().reference_stdout).is_file()
    assert Path(evaluated.result().reference_stderr).is_file()
    assert 'energy' in evaluated.result().info.keys()
    assert 'stress' in evaluated.result().info.keys()
    assert 'forces' in evaluated.result().arrays.keys()
    assert np.allclose(
            -1.165271567241256 / molmod.units.electronvolt,
            evaluated.result().info['energy'],
            )
    forces_reference = np.array([
            [-0.01215748,    0.00001210,    0.00001210],
            [ 0.01217855,    0.00001150,    0.00001150]])
    forces_reference /= molmod.units.electronvolt
    forces_reference *= molmod.units.angstrom
    assert np.allclose(
            forces_reference,
            evaluated.result().arrays['forces'],
            atol=1e-5,
            )
    stress_reference = -1.0 * np.array([
            [4.81505171868E-01,  4.49529611310E-06,  4.49529611348E-06],
            [4.49529611310E-06, -9.53484935396E-03,  1.47299106211E-04],
            [4.49529611348E-06,  1.47299106211E-04, -9.53484935396E-03]])
    stress_reference *= (1e9 * Pascal)
    assert np.allclose(
            stress_reference,
            evaluated.result().info['stress'],
            #atol=1e-5,
            )

    # check whether NullState evaluates to NullState
    state = cp2k_reference.evaluate(NullState)
    assert state.result() == NullState

    # check number of mpi processes
    with open(evaluated.result().reference_stdout, 'r') as f:
        content = f.read()
    context = psiflow.context()
    ncores = context[CP2KReference][0].ncores
    omp_num_threads = context[CP2KReference][0].omp_num_threads
    mpi_num_procs = ncores // omp_num_threads
    lines = content.split('\n')
    for line in lines:
        if 'Total number of message passing processes' in line:
            nprocesses = int(line.split()[-1])
        if 'Number of threads for this process' in line:
            nthreads = int(line.split()[-1])
    assert mpi_num_procs == nprocesses
    assert omp_num_threads == nthreads


def test_cp2k_failure(context, cp2k_data, tmp_path):
    cp2k_input = """
&FORCE_EVAL
   METHOD Quickstep
   STRESS_TENSOR ANALYTICAL
   &DFT
      UKS  F
      MULTIPLICITY  1
      BASIS_SET_FILE_NAME  dummy
      POTENTIAL_FILE_NAME  dummy
      &SCF
         MAX_SCF  10
         MAX_DIIS  8
         EPS_SCF  1.0E-01
         SCF_GUESS  RESTART
         &OT
            MINIMIZER  CG
            PRECONDITIONER  FULL_SINGLE_INVERSE
         &END OT
         &OUTER_SCF T
            MAX_SCF  10
            EPS_SCF  1.0E-01
         &END OUTER_SCF
      &END SCF
      &QS
         METHOD  GPW
         EPS_DEFAULT  1.0E-4
         EXTRAPOLATION  USE_GUESS
      &END QS
      &MGRID
         REL_CUTOFF [Ry]  60.0
         NGRIDS  5
         CUTOFF [Ry] 200
      &END MGRID
      &XC
         DENSITY_CUTOFF   1.0E-10
         GRADIENT_CUTOFF  1.0E-10
         TAU_CUTOFF       1.0E-10
         &XC_FUNCTIONAL PBE
         &END XC_FUNCTIONAL
         &VDW_POTENTIAL
            POTENTIAL_TYPE  PAIR_POTENTIAL
            &PAIR_POTENTIAL
               TYPE  DFTD3(BJ)
               PARAMETER_FILE_NAME  parameter
               REFERENCE_FUNCTIONAL PBE
               R_CUTOFF  25
            &END PAIR_POTENTIAL
         &END VDW_POTENTIAL
      &END XC
   &END DFT
   &SUBSYS
      &KIND H
         ELEMENT  H
         BASIS_SET XXXXXXXXXX
         POTENTIAL GTH-PBE-q1
      &END KIND
   &END SUBSYS
   &PRINT
      &STRESS_TENSOR ON
      &END STRESS_TENSOR
      &FORCES
      &END FORCES
   &END PRINT
&END FORCE_EVAL
""" # incorrect input file
    reference = CP2KReference(cp2k_input=cp2k_input)
    for key, value in cp2k_data.items():
        with open(tmp_path / key, 'w') as f:
            f.write(value)
        reference.add_file(key, tmp_path / key)
    atoms = FlowAtoms( # simple H2 at ~optimized interatomic distance
            numbers=np.ones(2),
            cell=5 * np.eye(3),
            positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
            pbc=True,
            )
    evaluated = reference.evaluate(atoms)
    assert isinstance(evaluated, AppFuture)
    assert evaluated.result().reference_status == False
    assert 'energy' not in evaluated.result().info.keys()
    with open(evaluated.result().reference_stdout, 'r') as f:
        log = f.read()
    assert 'ABORT' in log # verify error is captured
    assert 'requested basis set' in log


def test_cp2k_timeout(context, cp2k_reference):
    atoms = FlowAtoms( # simple H2 at ~optimized interatomic distance
            numbers=np.ones(2),
            cell=20 * np.eye(3), # box way too large
            positions=np.array([[0, 0, 0], [3, 0, 0]]),
            pbc=True,
            )
    evaluated = cp2k_reference.evaluate(atoms)
    assert isinstance(evaluated, AppFuture)
    assert evaluated.result().reference_status == False
    assert 'energy' not in evaluated.result().info.keys()


def test_emt_atomic_energies(context, dataset):
    reference = EMTReference()
    for element in ['H', 'Cu']:
        energy  = reference.compute_atomic_energy(element, box_size=5)
        energy_ = reference.compute_atomic_energy(element, box_size=7)
        assert energy.result() < energy_.result()


def test_cp2k_atomic_energies(cp2k_reference, dataset):
    element = 'H'
    energy  = cp2k_reference.compute_atomic_energy(element, box_size=8)
    assert abs(energy.result() - (-13.6)) < 1e-1
    # testing of additional atoms is possible but not really critical


def test_data_set_formation_energy(context, dataset):
    reference = EMTReference()
    nstates = dataset.length().result()
    energies = [dataset[i].result().info['energy'] for i in range(nstates)]
    dataset_ = dataset.set_formation_energy(
            H=reference.compute_atomic_energy('H', box_size=8),
            Cu=reference.compute_atomic_energy('Cu', box_size=8),
            )
    assert 'atomic_energy_H' in dataset_[0].result().info
    assert 'atomic_energy_Cu' in dataset_[0].result().info
    assert 'formation_energy' in dataset_.energy_labels().result()
    formation = [dataset_[i].result().info['formation_energy'] for i in range(nstates)]

    energy_Cu = reference.compute_atomic_energy('Cu', box_size=8).result()
    assert dataset_[0].result().info['atomic_energy_Cu'] == energy_Cu
    energy_H = reference.compute_atomic_energy('H', box_size=8).result()
    assert energy_Cu != 0
    assert energy_H  != 0
    assert np.allclose(
            formation[0],
            energies[0] - 3 * energy_Cu - energy_H,
            )
    data = dataset_.get(indices=[0, 1, 2])
    assert 'formation_energy' in data[0].result().info

    dataset__ = dataset.set_formation_energy(
            H=29,
            Cu=reference.compute_atomic_energy('Cu', box_size=8),
            )
    assert np.allclose( # will check absolute energies
            Dataset.get_errors(dataset__, dataset, properties=['energy']).result()[:, 0],
            np.zeros(dataset.length().result()),
            )
    assert not np.allclose( # will check formation energies
            Dataset.get_errors(dataset__, dataset_, properties=['energy']).result()[:, 0],
            np.zeros(dataset.length().result()),
            )
    dataset__ = dataset.set_formation_energy(
            H=reference.compute_atomic_energy('H', box_size=8),
            Cu=reference.compute_atomic_energy('Cu', box_size=8),
            )
    assert np.allclose( # will check formation energies; now OK
            Dataset.get_errors(dataset__, dataset_, properties=['energy']).result()[:, 0],
            np.zeros(dataset.length().result()),
            )

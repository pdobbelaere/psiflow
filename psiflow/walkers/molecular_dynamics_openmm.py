from sys import stdout
from pathlib import Path
import numpy as np
import argparse
import shutil
import torch

import mdtraj
import openmm
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential
from openmmplumed import PlumedForce

from ase.io import read
from ase.data import chemical_symbols
import ase.build
import ase.io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--ncores', default=None, type=int)
    parser.add_argument('--dtype', default=None, type=str)
    parser.add_argument('--atoms', default=None, type=str)

    # pars
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--timestep', default=None, type=float)
    parser.add_argument('--steps', default=None, type=int)
    parser.add_argument('--step', default=None, type=int)
    parser.add_argument('--start', default=None, type=int)
    parser.add_argument('--temperature', default=None, type=float)
    parser.add_argument('--pressure', default=None, type=float)
    parser.add_argument('--force_threshold', default=None, type=float)
    parser.add_argument('--initial_temperature', default=None, type=float)

    parser.add_argument('--model-cls', default=None, type=str) # model name
    parser.add_argument('--model', default=None, type=str) # model name
    parser.add_argument('--keep-trajectory', default=None, type=bool)
    parser.add_argument('--trajectory', default=None, type=str)
    parser.add_argument('--walltime', default=None, type=float)

    args = parser.parse_args()

    path_plumed = 'plumed.dat'

    assert args.device in ['cpu', 'cuda']
    assert args.dtype in ['float32', 'float64']
    assert Path(args.atoms).is_file()
    assert args.model_cls in ['MACEModel', 'NequIPModel', 'AllegroModel']
    assert Path(args.model).is_file()
    assert args.trajectory is not None

    import signal
    class TimeoutException(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutException

    signal.signal(signal.SIGTERM, timeout_handler)

    print('torch: initial num threads: ', torch.get_num_threads())
    torch.set_num_threads(args.ncores)
    print('torch: num threads set to ', torch.get_num_threads())
    if args.dtype == 'float64':
        print('simulation will proceed in float32, even though float64 was requested')
    if args.force_threshold is not None:
        print('IGNORING requested force threshold at {} eV/A!'.format(args.force_threshold))
        print('force thresholds are only supported in the yaff engine')
    torch.set_default_dtype(torch.float32)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    atoms = read(args.atoms)
    initial = atoms.copy()

    topology = openmm.app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue('bla', chain)
    for i in range(len(atoms)):
        symbol = chemical_symbols[atoms.numbers[i]]
        element = app.Element.getBySymbol(symbol)
        topology.addAtom(symbol, element, residue)
    if atoms.pbc.all():
        cell = np.array(atoms.cell[:]) * 0.1 # A -> nm
        topology.setPeriodicBoxVectors(cell)
    positions = atoms.positions * 0.1 # A->nm

    A_to_nm          = 0.1
    eV_to_kJ_per_mol = 96.49
    if args.model_cls == 'MACEModel':
        model_cls = 'mace'
    elif args.model_cls in ['NequIPModel', 'AllegroModel']:
        model_cls = 'nequip'
    potential = MLPotential(
            model_cls,
            model_path=args.model, 
            distance_to_nm=A_to_nm, 
            energy_to_kJ_per_mol=eV_to_kJ_per_mol,
            device=args.device,
            )
    system = potential.createSystem(topology, dtype='float32')

    temperature = args.temperature * unit.kelvin
    integrator = openmm.LangevinIntegrator(temperature, 1.0/unit.picoseconds, args.timestep * unit.femtosecond)
    if args.device == 'cuda':
        platform_name = 'CUDA'
    else:
        platform_name = 'CPU'
    platform = openmm.Platform.getPlatformByName(platform_name)
    simulation = app.Simulation(topology, system, integrator, platform=platform)
    simulation.context.setPositions(positions)

    if Path('plumed.dat').is_file():
        try_manual_plumed_linking()
        with open('plumed.dat', 'r') as f:
            plumed_input = f.read()
        system.addForce(PlumedForce(plumed_input))

    if args.pressure is None:
        print('sampling NVT ensemble ...')
    else:
        print('sampling NPT ensemble ...')
        barostat = openmm.MonteCarloFlexibleBarostat(
                10 * args.pressure,          # to bar
                args.temperature,
                scaleMoleculesAsRigid=False, # ensure correct pressure control for materials
                )
        system.addForce(barostat)
    simulation.context.reinitialize(preserveState=True)


    #simulation.reporters.append(app.PDBReporter('output.pdb', 50, enforcePeriodicBox=True))
    hdf = mdtraj.reporters.HDF5Reporter(
            args.trajectory,
            reportInterval=args.step,
            coordinates=True,
            time=False,
            cell=True,
            potentialEnergy=False,
            temperature=False,
            velocities=True,
            )
    log = app.StateDataReporter(
            stdout,
            args.step,
            step=True,
            potentialEnergy=True,
            temperature=True,
            volume=True,
            elapsedTime=True,
            )
    simulation.reporters.append(hdf)
    simulation.reporters.append(log)

    try:
        simulation.step(100000)
        print('current step: {}'.format(simulation.currentStep))
    except TimeoutException:
        print('current step: {}'.format(simulation.currentStep))
        hdf.close()

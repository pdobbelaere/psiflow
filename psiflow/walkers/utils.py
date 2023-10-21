import molmod
import numpy as np
import yaff
from ase import Atoms
from ase.geometry import Cell
from ase.io import write
from scipy.stats import chi2


class ForcePartPlumed(yaff.external.ForcePartPlumed):
    """Remove timer from _internal_compute to avoid pathological errors"""

    def _internal_compute(self, gpos, vtens):
        self.plumed.cmd("setStep", self.plumedstep)
        self.plumed.cmd("setPositions", self.system.pos)
        self.plumed.cmd("setMasses", self.system.masses)
        if self.system.charges is not None:
            self.plumed.cmd("setCharges", self.system.charges)
        if self.system.cell.nvec > 0:
            rvecs = self.system.cell.rvecs.copy()
            self.plumed.cmd("setBox", rvecs)
        # PLUMED always needs arrays to write forces and virial to, so
        # provide dummy arrays if Yaff does not provide them
        # Note that gpos and forces differ by a minus sign, which has to be
        # corrected for when interacting with PLUMED
        if gpos is None:
            my_gpos = np.zeros(self.system.pos.shape)
        else:
            gpos[:] *= -1.0
            my_gpos = gpos
        self.plumed.cmd("setForces", my_gpos)
        if vtens is None:
            my_vtens = np.zeros((3, 3))
        else:
            my_vtens = vtens
        self.plumed.cmd("setVirial", my_vtens)
        # Do the actual calculation, without an update; this should
        # only be done at the end of a time step
        self.plumed.cmd("prepareCalc")
        self.plumed.cmd("performCalcNoUpdate")
        if gpos is not None:
            gpos[:] *= -1.0
        # Retrieve biasing energy
        energy = np.zeros((1,))
        self.plumed.cmd("getBias", energy)
        return energy[0]


class ForceThresholdExceededException(Exception):
    pass


class ForcePartASE(yaff.pes.ForcePart):
    """YAFF Wrapper around an ASE calculator"""

    def __init__(self, system, atoms):
        """Constructor

        Parameters
        ----------

        system : yaff.System
            system object

        atoms : ase.Atoms
            atoms object with calculator included.

        force_threshold : float [eV/A]

        """
        yaff.pes.ForcePart.__init__(self, "ase", system)
        self.system = system  # store system to obtain current pos and box
        self.atoms = atoms

    def _internal_compute(self, gpos=None, vtens=None):
        self.atoms.set_positions(self.system.pos / molmod.units.angstrom)
        if self.atoms.pbc.all():
            self.atoms.set_cell(
                Cell(self.system.cell._get_rvecs() / molmod.units.angstrom)
            )
        energy = self.atoms.get_potential_energy() * molmod.units.electronvolt
        if gpos is not None:
            forces = self.atoms.get_forces()
            gpos[:] = -forces * molmod.units.electronvolt / molmod.units.angstrom
        if vtens is not None:
            if self.atoms.pbc.all():
                stress = self.atoms.get_stress(voigt=False)
                volume = np.linalg.det(self.atoms.get_cell())
                vtens[:] = volume * stress * molmod.units.electronvolt
            else:
                vtens[:] = 0.0
        return energy


class ForceField(yaff.pes.ForceField):
    """Implements force threshold check"""

    def __init__(self, *args, force_threshold=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_threshold = force_threshold

    def _internal_compute(self, gpos, vtens):
        if self.needs_nlist_update:  # never necessary?
            self.nlist.update()
            self.needs_nlist_update = False
        result = sum([part.compute(gpos, vtens) for part in self.parts])
        forces = (-1.0) / molmod.units.electronvolt * molmod.units.angstrom * gpos
        self.check_threshold(forces)
        return result

    def check_threshold(self, forces):
        max_force = np.max(np.linalg.norm(forces, axis=1))
        index = np.argmax(np.linalg.norm(forces, axis=1))
        if max_force > self.force_threshold:
            raise ForceThresholdExceededException(
                "Max force exceeded: {} eV/A by atom index {}".format(max_force, index),
            )


def create_forcefield(atoms, force_threshold):
    """Creates force field from ASE atoms instance"""
    if atoms.pbc.all():
        rvecs = atoms.get_cell() * molmod.units.angstrom
    else:
        rvecs = None
    system = yaff.System(
        numbers=atoms.get_atomic_numbers(),
        pos=atoms.get_positions() * molmod.units.angstrom,
        rvecs=rvecs,
    )
    system.set_standard_masses()
    part_ase = ForcePartASE(system, atoms)
    return ForceField(system, [part_ase], force_threshold=force_threshold)


class DataHook(yaff.VerletHook):
    def __init__(self, start=0, step=1):
        super().__init__(start, step)
        self.atoms = None
        self.data = []

    def init(self, iterative):
        if iterative.ff.system.cell.nvec > 0:
            cell = iterative.ff.system.cell._get_rvecs() / molmod.units.angstrom
        else:
            cell = None
        self.atoms = Atoms(
            numbers=iterative.ff.system.numbers.copy(),
            positions=iterative.ff.system.pos / molmod.units.angstrom,
            cell=cell,
            pbc=cell is not None,
        )

    def pre(self, iterative):
        pass

    def post(self, iterative):
        pass

    def __call__(self, iterative):
        self.atoms.set_positions(iterative.ff.system.pos / molmod.units.angstrom)
        if self.atoms.pbc.all():
            self.atoms.set_cell(
                iterative.ff.system.cell._get_rvecs() / molmod.units.angstrom
            )
        self.data.append(self.atoms.copy())


class ExtXYZHook(yaff.VerletHook):  # xyz file writer; obsolete
    def __init__(self, path_xyz, start=0, step=1):
        super().__init__(start, step)
        self.path_xyz = path_xyz
        self.atoms = None
        self.nwrites = 0
        self.temperatures = []

    def init(self, iterative):
        if iterative.ff.system.cell.nvec > 0:
            cell = iterative.ff.system.cell._get_rvecs() / molmod.units.angstrom
        else:
            cell = None
        self.atoms = Atoms(
            numbers=iterative.ff.system.numbers.copy(),
            positions=iterative.ff.system.pos / molmod.units.angstrom,
            cell=cell,
            pbc=cell is not None,
        )

    def pre(self, iterative):
        pass

    def post(self, iterative):
        pass

    def __call__(self, iterative):
        if iterative.counter > 0:  # first write is manual
            self.atoms.set_positions(iterative.ff.system.pos / molmod.units.angstrom)
            if self.atoms.pbc.all():
                self.atoms.set_cell(
                    iterative.ff.system.cell._get_rvecs() / molmod.units.angstrom
                )
            write(self.path_xyz, self.atoms, append=True)
            self.nwrites += 1
            self.temperatures.append(iterative.temp)


def apply_strain(strain, box0):
    """Applies a strain tensor to a reference box

    The resulting strained box matrix is obtained based on:

        box = box0 @ sqrt(2 * strain + I)

    where the second argument is computed based on a diagonalization of
    2 * strain + I.

    Parameters
    ----------

    strain : ndarray of shape (3, 3)
        desired strain matrix

    box0 : ndarray of shape (3, 3)
        reference box matrix

    """
    assert np.allclose(strain, strain.T)
    A = 2 * strain + np.eye(3)
    values, vectors = np.linalg.eigh(A)
    sqrtA = vectors @ np.sqrt(np.diag(values)) @ vectors.T
    box = box0 @ sqrtA
    return box


def compute_strain(box, box0):
    """Computes the strain of a given box with respect to a reference

    The strain matrix is defined by the following expression

        strain = 0.5 * (inv(box0) @ box @ box.T @ inv(box0).T - I)

    Parameters
    ----------

    box : ndarray of shape (3, 3)
        box matrix for which to compute the strain

    box0 : ndarray of shape (3, 3)
        reference box matrix

    """
    box0inv = np.linalg.inv(box0)
    return 0.5 * (box0inv @ box @ box.T @ box0inv.T - np.eye(3))


def parse_yaff_output(stdout):
    temperatures = []
    counter = 0
    time = 0
    for line in stdout.split("\n"):
        if "VERLET" in line:
            try:
                a = [float(s) for s in line.split()[1:]]
            except ValueError:
                continue
            temperatures.append(float(line.split()[3]))
            counter = int(line.split()[1])
            time = float(line.split()[6])
        else:
            pass
    if len(temperatures) == 0:
        temperatures.append(-1)
    return counter, np.mean(np.array(temperatures)), time


def parse_openmm_output(stdout):
    temperatures = []
    counter = 0
    time = 0
    start = False
    for line in stdout.split("\n"):
        if start:
            try:
                metrics = [s for s in line.split(",")]
                time = float(metrics[-1])
                counter = int(metrics[0])
                temperatures.append(float(metrics[2]))
            except ValueError:
                break
        if '#"Step","Potential Energy (kJ/mole)"' in line:
            start = True
    if len(temperatures) == 0:
        temperatures.append(-1)
    return counter, np.mean(np.array(temperatures)), time


def max_temperature(temperature: float, natoms: int, quantile: float) -> float:
    ndof = 3 * natoms
    return chi2.ppf(1 - quantile, ndof) * temperature / ndof


def max_temperature_std(temperature: float, natoms: int, N: float) -> float:
    return temperature + N * temperature / np.sqrt(3 * natoms)


def get_velocities_at_temperature(temperature, masses):
    from ase.units import kB

    velocities = np.random.normal(0, 1, (len(masses), 3))
    velocities *= np.sqrt(kB * temperature / masses).reshape(-1, 1)
    actual = (velocities**2 * masses.reshape(-1, 1)).mean() / kB
    scale = np.sqrt(temperature / actual)
    velocities *= scale
    return velocities

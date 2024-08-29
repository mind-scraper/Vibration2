""" 
###Vibrations2###
This class performing the vibrational calculation twice:
1st: Calculating Hessian (H) in cartesian coordinate (x, y, z) using ASE vibrations module
2nd: Refining the H in the normal coordinate (qi)
In both case H is approximated using finite difference methods. 

Many part of the code are copied from ASE. 
https://databases.fysik.dtu.dk/ase/ase/vibrations/vibrations.html

Written by
Samuel Eka Putra Payong Masan
Ph.D. student at Morikawa Group
Osaka University
August 2024
######

Input parameter

    #Following inputs are the same as in ASE's vibration
    atoms: Atoms object
        The atoms to work on.

    indices: list of int
        default=None
        List of indices of atoms to vibrate.  Default behavior is
        to vibrate all atoms.

    name: str
        default="vib"
        Name to use for folder containig forces files for 1st calculation.

    delta: float
        default=0.01
        Magnitude of displacements.

    nfree: int
        default=2
        Number of displacements per atom and cartesian coordinate, 2 and 4 are
        supported. Default is 2 which will displace each atom +delta and
        -delta for each cartesian coordinate.

        
    #Following inputs are NOT exist in ASE's vibration
    isolated: bool
        default=False
        Wether the system is an isolated molecule (True) or not (False).
    
        if isolated=True please also specify following input
            mol_shape: str
                default='nonlinear'
                The shape of molecule. It is either 'linear' or 'nonlinear'

    name2: str
        default="vib2"
        Name to use for folder containig forces files for 1st calculation.

    FD_method: str
        default="forces"
        "forces": Use finite difference of forces to calculate H
                    H[i] =  (f_min_q - f_plus_q)/(2*disp) 
        "energy": Use finite difference of energy to calculate H
                    H[i,i] = (e_min + e_plus - 2 * e_eq)/(disp)**2
        Here disp is the displacement magnitude. 

    fmax: float
        default=2e-3 Ry/Bohr
        Target maximum force acting on the displaced atoms. 
        This is used to determine the displacement magnitude.
        see e.q. https://doi.org/10.1103/PhysRevB.110.075409

Example
    >>> from ase.optimize import BFGS as relaxer
    >>> from ase import Atoms
    >>> from ase.calculators.emt import EMT
    >>> from vibrations2 import Vibrations2
    >>> atoms = Atoms('N2', [(0, 0, 0), (0, 0, 1.1)], calculator=EMT())
    >>> opt = relaxer(atoms)
    >>> opt.run(fmax=0.01)
          Step     Time          Energy         fmax
    BFGS:    0 16:38:17        0.440344        3.2518
    BFGS:    1 16:38:17        0.264361        0.3475
    BFGS:    2 16:38:17        0.262860        0.0805
    BFGS:    3 16:38:17        0.262777        0.0015
    >>> vib = Vibrations2(atoms, isolated=True, mol_shape='linear')
    >>> vib.run()
    Summary of the vibrational analysis using 
    displacement in cartesian coordinate (x, y, z).

    ---------------------
    #    meV     cm^-1
    ---------------------
    0    0.0       0.0
    1    0.0       0.0
    2    0.0       0.0
    3    1.4      11.5
    4    1.4      11.5
    5  152.7    1231.3
    ---------------------
    Zero-point energy: 0.078 eV

    Calcualting the vibrational analysis using 
    displacement in normal coordinate (Qi) . . . 

    Finished. Print the summarry with vib.summary()
    >>> vib.summary()
    ---------------------
    #    meV     cm^-1
    ---------------------
    0    0.0       0.0
    1    0.0       0.0
    2    0.0       0.0
    3    0.0       0.0
    4    0.0       0.0
    5  152.7    1231.2
    ---------------------
    Zero-point energy: 0.076 eV
    >>> vib.write_mode(-1)  # write last mode to trajectory file

Use with care ^_^
"""

import numpy as np
from ase.vibrations import Vibrations
from ase.vibrations.data import VibrationsData
from typing import Any, Dict, Iterator, List, Sequence, Tuple, TypeVar, Union
from ase.io import read
from ase import units
from ase import Atoms
import ase
import os
import sys
from itertools import combinations

class Vibrations2():
    def __init__(self, atoms, 
                 indices=None, 
                 name='vib', 
                 delta=0.01, 
                 nfree=2, 
                 isolated=False, 
                 mol_shape='nonlinear', 
                 name2='vib2', 
                 FD_method='forces', 
                 fmax=2e-3*units.Ry/units.Bohr,
                 ):

        self.atoms = atoms
        self.calc = atoms.calc
        if indices == None:
            self.indices = [atom.index for atom in self.atoms]
        else:
            self.indices = indices
        self.name = name
        self.name2 = name2
        self.delta = delta
        self.nfree = nfree
        self.isolated = isolated
        self.mol_shape = mol_shape
        self.FD_method = FD_method
        self.fmax = fmax
    
    def do_vib1(self):
        """
        Perform vibrational analysis with ASE.
        The resulting normal mode and frequency will be used for 
        the second vibrational analysis. 
        """
        vib = Vibrations(self.atoms, 
                         indices=self.indices, 
                         delta=self.delta,
                         name = self.name,
                         nfree = self.nfree,
                         )
        vib.run()
        print('Summary of the vibrational analysis using \n displacement in cartesian coordinate (x, y, z).\n')
        vib.summary()
        print('\nCalcualting the vibrational analysis using \n displacement in normal coordinate (Qi) . . . \n')
        return vib
        
    def run(self):
        """
        Main part of the code.
        """

        # Getting the frequencies and normal modes
        vib1 = self.do_vib1()
        energies = vib1.get_energies()
        modes = [vib1.get_mode(i) for i in range(len(energies))]
        modes = np.array(modes)

        # Calculate reduced mass for each modes
        mu = self.calculate_reduced_mass(modes)

        # Determining the number of modes

        if self.isolated:
            """
            For isolated molecules, the displacement will not be made in the
            rotational and translational mode.
            """
            Natoms = len(self.atoms)
            if self.mol_shape == 'linear':
                Nmode = 3*Natoms - 5
            elif self.mol_shape == 'nonlinear':
                Nmode = 3*Natoms - 6
        else:
            Nmode = len(modes)

        # Make directory for storing the displacement files
        try:
            os.mkdir(self.name2)    
        except:
            pass
        os.chdir(self.name2)
        
        # Calculate the forces of the displaced atoms along normal coordinate qi, then calculate the H(qi,qj)

        H = np.zeros((len(modes), len(modes))) 
        i = len(modes) - 1
        e_eq = self.atoms.get_potential_energy()
        for _ in range(Nmode):
            factor = self.get_amplitude(energies[i], mu[i])
            displacement = factor*modes[i]       

            # Get the transformation matrix from the normal modes. This matrix transform  (x, y, z) to qi
            Transform = np.array([mode[self.indices] for mode in modes])
            Transform = Transform.reshape((len(modes), len(modes)))

            # Copy the atomic structure and make displacement in +Q direction 
            try:
                atoms_displaced_plus = read(f'disp{i}_plus.xyz')
            except:
                atoms_displaced_plus = self.atoms.copy()
                atoms_displaced_plus.positions += displacement
                atoms_displaced_plus.calc = self.calc
                atoms_displaced_plus.get_potential_energy()
                atoms_displaced_plus.write(f'disp{i}_plus.xyz')            
            e_plus = atoms_displaced_plus.get_potential_energy()
            f_plus =  atoms_displaced_plus.get_forces()
            f_plus = f_plus[self.indices].reshape(1, 3*len(self.indices))
                                   

            # Copy the atomic structure and make displacement in -Q direction 
            try:
                atoms_displaced_min = read(f'disp{i}_min.xyz')
            except:
                atoms_displaced_min = self.atoms.copy()
                atoms_displaced_min.positions -= displacement
                atoms_displaced_min.calc = self.calc
                atoms_displaced_min.get_potential_energy()
                atoms_displaced_min.write(f'disp{i}_min.xyz')
            e_min = atoms_displaced_min.get_potential_energy()
            f_min = atoms_displaced_min.get_forces()            
            f_min = f_min[self.indices].reshape(1, 3*len(self.indices))
            
            # Transform forces from (x, y, z) to qi
            f_plus_q = np.dot(f_plus, Transform.T)
            f_min_q = np.dot(f_min, Transform.T)

                
            if self.FD_method == 'energy':
                # Probably better in avoiding eggbox effect?                
                H[i,i] = 0.5*(e_min + e_plus - 2 * e_eq)/(factor)**2
            elif self.FD_method == 'forces':
                H[i] =  0.5*(f_min_q - f_plus_q)/(2*factor) 
            #H supposed to be symmetric. Factor 0.5 is used to reduce the nummerical noise. 
            #The 1/2H is then added to its transpose 1/2H.T.

            i -= 1
        
        H += H.T #This H is in Q space thus no need to be wighted with mass. 

        omega2, vectors = np.linalg.eigh(H)

        unit_conversion = units._hbar * units.m / np.sqrt(units._e * units._amu)
        energies = unit_conversion * omega2.astype(complex)**0.5

        masses = self.atoms.get_masses()
        n_atoms = len(self.atoms)
        new_modes = vectors.T.reshape(n_atoms * 3, n_atoms, 3)
        new_modes = new_modes * masses[np.newaxis, :, np.newaxis]**-0.5
    
        self.energies = energies
        self.new_modes = modes
        os.chdir('..')
        print('\nFinished. Print the summarry with vib.summary()\n')

    def get_amplitude(self, e_vib, mu):       
        """
        Calculate the amplitude of the displacement along the Q. 
        The amplitude expect to causses the atomic structure to 
        feel maximum forces of fmax.
        """
        e_vib = e_vib.real + e_vib.imag  
        k = (e_vib*2*np.pi)**2 * mu
        amp = self.fmax / k**0.5
        return amp
        
    def summary(self):
        summary_lines = VibrationsData._tabulate_from_energies(self.energies)
        log_text = '\n'.join(summary_lines) + '\n'
        log=sys.stdout
        log.write(log_text)

    def write_mode(self, n=None, kT=units.kB * 300, nimages=30):
        """Write mode number n to trajectory file. If n is not specified,
        writes all non-zero modes."""
        if n is None:
            for index, energy in enumerate(self.energies):
                if abs(energy) > 1e-5:
                    self.write_mode(n=index, kT=kT, nimages=nimages)
            return

        else:
            n %= len(self.energies)

        with ase.io.Trajectory('%s.%d.traj' % (self.name2, n), 'w') as traj:
            for image in (self.iter_animated_mode(n,
                                              temperature=kT, frames=nimages)):
                traj.write(image)

    def iter_animated_mode(self, mode_index: int,
                           temperature: float = units.kB * 300,
                           frames: int = 30) -> Iterator[Atoms]:
        """Obtain animated mode as a series of Atoms

        Args:
            mode_index: Selection of mode to animate
            temperature: In energy units - use units.kB * T_IN_KELVIN
            frames: number of image frames in animation

        Yields:
            Displaced atoms following vibrational mode

        """

        mode = (self.new_modes[mode_index]
                * np.sqrt(temperature / abs(self.energies[mode_index])))

        for phase in np.linspace(0, 2 * np.pi, frames, endpoint=False):
            atoms = self.atoms.copy()
            atoms.positions += np.sin(phase) * mode

            yield atoms

    def calculate_reduced_mass(self, displacement_vectors):
        masses = self.atoms.get_masses()
        reduced_masses = []

        for displacement in displacement_vectors:
            displacement = np.array(displacement)
            norm = np.linalg.norm(displacement)
            if norm == 0:
                reduced_masses.append(0)
                continue

            displacement_unit = displacement / norm
            inv_mass_sum = 0

            for i, mass in enumerate(masses):
                inv_mass_sum += np.dot(displacement_unit.flatten(), displacement_unit.flatten()) / mass

            reduced_mass = 1 / inv_mass_sum
            reduced_masses.append(reduced_mass)

        return reduced_masses
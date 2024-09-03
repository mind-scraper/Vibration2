""" 
###Vibrations2###
This class performing the vibrational calculation with the following steps.
1st: Calculating Hessian (H) in cartesian coordinate (x, y, z) using ASE vibrations module
2nd: Refining the eigen value (omega2) in the normal coordinate (qi)
3rd: Repeat 2nd until error < error_thr

The derivative of forces is approximated using finite difference methods. 

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
                default='None'
                The shape of molecule. It is either 'linear' or 'nonlinear'

    name2: str
        default="vib2"
        Name to use for folder containig forces files for 1st calculation.

    FD_method: str
        default="forces"
        "forces": Use finite difference of forces
                    omega2 =  (f_min_q - f_plus_q)/(2*disp) 
        "energy": Use finite difference of energy to calculate H. This is just experimental. Not recommended to use. 
                    omega2 = (e_min + e_plus - 2 * e_eq)/(disp)**2

    fmax: float
        default=2e-3 Ry/Bohr
        Target maximum force acting on the displaced atoms. 
        This is used to determine the displacement magnitude.
        see e.q. https://doi.org/10.1103/PhysRevB.110.075409

    error_thr: float
        default=10 cm^-1
        Threshold to stop the iteration. It is definied as the frequency difference between two consecutive iteration. 

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
    Summary of the vibrational analysis by 
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

    Calcualting the vibrational analysis by 
    displacement in normal coordinate (Qi) . . . 
    
    ### Refining mode 5 ###
    Initial nu= 1231.26+0.00j cm^-1
    New nu = 1231.30+0.00j cm^-1
    Error = 0.04 cm^-1
    
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
    5  152.7    1231.3
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
from copy import deepcopy, copy

class Vibrations2():
    def __init__(self, atoms, 
                 indices=None, 
                 name='vib', 
                 delta=0.01, 
                 nfree=2, 
                 isolated=False, 
                 mol_shape=None, 
                 name2='vib2', 
                 FD_method='forces', 
                 fmax=3e-3*units.Ry/units.Bohr,
                 error_thr = 10, #Error in cm^-1
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

        if len(self.atoms) == len(self.indices):
            if self.isolated:
                if self.mol_shape is None:
                    raise ValueError("Please set the molecule shape for isolated molecules.")
            else:
                raise ValueError("Isolated system detected. Please set isolated=True and specify the molecule shape.")
        
        self.FD_method = FD_method
        self.fmax = fmax
        self.error_thr = error_thr

        # Determine the number of modes for the refinement
        if self.isolated:
            """
            For isolated molecules, the displacement will not be made in the
            rotational and translational mode.
            """
            if self.mol_shape == 'linear':
                Nmode = 3*len(self.indices) - 5
            elif self.mol_shape == 'nonlinear':
                Nmode = 3*len(self.indices) - 6
        else:
            Nmode = 3*len(self.indices)      
        self.Nmode = Nmode
    
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
        
        print('Summary of the vibrational analysis by \n displacement in cartesian coordinate (x, y, z).\n')
        vib.summary()
        
        print('\nCalcualting the vibrational analysis by \n displacement in normal coordinate (Qi) . . . \n')
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

        # Make directory for storing the displacement files
        try:
            os.mkdir(self.name2)    
        except:
            pass
        
        # Calculate the forces of the displaced atoms along normal coordinate qi, then calculate omega2
                
        energies_new = np.zeros(len(modes) , dtype=complex)
        self.all_delta = np.zeros(len(modes))
        i = 3*len(self.indices) - 1 #Index for iteration
        for _ in range(self.Nmode):
            print(f'\n### Refining mode {i} ###')
            freq = energies[i]/units.invcm
            print(f'Initial nu= {freq:.2f} cm^-1')
            error = self.error_thr + 1 #Dummy to set the initial error higer than threshold            
            factor = None
            while error > self.error_thr:                        
                if factor is not None:
                    factor = (factor + self.get_amplitude(energies[i], mu[i]))/2
                    if factor>1:
                        factor=1
                else:
                    factor = self.get_amplitude(energies[i], mu[i])

                displacement = factor*modes[i]

                # Copy the atomic structure and make displacement in +Q direction 
                try:
                    atoms_displaced_plus = read(f'{self.name2}/disp{i}_plus.xyz')
                    os.remove(f'{self.name2}/disp{i}_plus.xyz')
                    restart = True
                except:                        
                    atoms_displaced_plus = self.atoms.copy()
                    atoms_displaced_plus.positions += displacement
                    atoms_displaced_plus.calc = copy(self.calc)
                    restart = False
                e_plus = atoms_displaced_plus.get_potential_energy()
                f_plus = atoms_displaced_plus.get_forces()

                # Copy the atomic structure and make displacement in -Q direction 
                try:
                    atoms_displaced_min = read(f'{self.name2}/disp{i}_min.xyz')
                    os.remove(f'{self.name2}/disp{i}_min.xyz')
                except:    
                    atoms_displaced_min = self.atoms.copy()
                    atoms_displaced_min.positions -= displacement
                    atoms_displaced_min.calc = copy(self.calc)
                e_min = atoms_displaced_min.get_potential_energy()
                f_min = atoms_displaced_min.get_forces()

                #Take dot product between forces and the q
                u = modes[i]
                f_plus_q = f_plus[self.indices].ravel() @ u[self.indices].ravel().T
                f_min_q = f_min[self.indices].ravel() @ u[self.indices].ravel().T 

                #Calculate the displacement magnitude. 
                # It is the same as prev. variable "factor" for a fresh start calculation. 
                # But it is different for restarted calculation. 
                # That's why I calculate it manually from displacement of the atoms from equilibirum position. 
                disp_pos = atoms_displaced_plus.get_positions()
                eq_pos = self.atoms.get_positions()
                delta = disp_pos - eq_pos
                delta = (delta[self.indices].ravel() @ delta[self.indices].ravel().T)**0.5
                u  = (u[self.indices].ravel() @ u[self.indices].ravel().T)**0.5
                delta = delta / u

                if self.FD_method == 'energy':
                    # Probably better in avoiding eggbox effect?
                    e_eq = self.atoms.get_potential_energy()                    
                    omega2 = (e_min + e_plus - 2 * e_eq)/(delta)**2
                elif self.FD_method == 'forces':
                    omega2 =  (f_min_q - f_plus_q)/(2*delta)    

                unit_conversion = units._hbar * units.m / np.sqrt(units._e * units._amu)
                energies_new[i] = unit_conversion * omega2.astype(complex)**0.5               
                 
                print(f'New nu = {energies_new[i]/units.invcm:.2f} cm^-1')            
                
                if restart:
                    error = 0
                    energies[i] = energies_new[i]
                else:                    
                    error = self.get_error(energies[i], energies_new[i])
                    print(f'Error = {error:.2f} cm^-1')
                    energies[i] = energies_new[i]                            

            #Write converged coordinate
            atoms_displaced_plus.write(f'{self.name2}/disp{i}_plus.xyz')   
            atoms_displaced_min.write(f'{self.name2}/disp{i}_min.xyz')

            self.all_delta[i] = delta
            i -= 1

        self.energies = energies_new
        self.modes = modes
        print('\nFinished. Print the summary with vib.summary()\n')

    def get_error(self, e_new, e_old):
        error = (abs(e_new.real - e_old.real) + abs(e_new.imag - e_old.imag)) / units.invcm
        return error

    def get_amplitude(self, e_vib, mu):  
        """
        Calculate the amplitude of the displacement along the Q. 
        The amplitude expected to causses the atomic structure to 
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

        with ase.io.Trajectory('%s.%d.traj' % (self.name, n), 'w') as traj:
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

        mode = (self.modes[mode_index]
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
    
    def write_disp(self):
        """
        Write the displacement vectors as xyz files.
        """

        dir = 'disp'
        
        try:
            os.mkdir(dir)
        except:
            pass        

        pos_eq = self.atoms.get_positions()
        i = 3*len(self.indices) - 1
        for _ in range(self.Nmode):
            atoms_displaced = read(f'{self.name2}/disp{i}_plus.xyz')            
            pos_disp = atoms_displaced.get_positions()
            disp = pos_disp - pos_eq
            atoms_disp_only = self.atoms.copy()
            atoms_disp_only.set_positions(disp, apply_constraint=False)
            atoms_disp_only.write(f'{dir}/disp{i}.xyz')

            i -= 1

        np.savetxt(f'{dir}/all_delta.txt', self.all_delta)

    def run_from_disp(self):
        """
        One can directly calculate the eigen value if the displacement vectors is present inside the disp directory.
        """        

        # Make directory for storing the displacement files
        try:
            os.mkdir(self.name2)    
        except:
            pass
        
        # Calculate the forces of the displaced atoms along normal coordinate qi, then calculate omega2

        energies_new = np.zeros(3*len(self.indices) , dtype=complex)
        i =  3*len(self.indices) - 1
        
        all_delta = np.loadtxt('disp/all_delta.txt')
        
        for _ in range(self.Nmode):
            displacement = read(f'disp/disp{i}.xyz')
            displacement = displacement.get_positions()

            # Copy the atomic structure and make displacement in +Q direction 
            try:
                atoms_displaced_plus = read(f'{self.name2}/disp{i}_plus.xyz')
            except:                        
                atoms_displaced_plus = self.atoms.copy()
                atoms_displaced_plus.positions += displacement
                atoms_displaced_plus.calc = copy(self.calc)
            e_plus = atoms_displaced_plus.get_potential_energy()
            f_plus = atoms_displaced_plus.get_forces()

            # Copy the atomic structure and make displacement in -Q direction 
            try:
                atoms_displaced_min = read(f'{self.name2}/disp{i}_min.xyz')
            except:    
                atoms_displaced_min = self.atoms.copy()
                atoms_displaced_min.positions -= displacement
                atoms_displaced_min.calc = copy(self.calc)
            e_min = atoms_displaced_min.get_potential_energy()
            f_min = atoms_displaced_min.get_forces()
            
            delta = all_delta[i]
            #Take dot product between forces and the q
            u = displacement/delta
            f_plus_q = f_plus[self.indices].ravel() @ u[self.indices].ravel().T
            f_min_q = f_min[self.indices].ravel() @ u[self.indices].ravel().T             

            if self.FD_method == 'energy':                
                # Probably better in avoiding eggbox effect?
                e_eq = self.atoms.get_potential_energy()
                omega2 = (e_min + e_plus - 2 * e_eq)/(delta)**2
            elif self.FD_method == 'forces':
                omega2 =  (f_min_q - f_plus_q)/(2*delta)    

            unit_conversion = units._hbar * units.m / np.sqrt(units._e * units._amu)
            energies_new[i] = unit_conversion * omega2.astype(complex)**0.5                                                                    

            #Write converged coordinate
            atoms_displaced_plus.write(f'{self.name2}/disp{i}_plus.xyz')   
            atoms_displaced_min.write(f'{self.name2}/disp{i}_min.xyz')
            i -= 1

        self.energies = energies_new
        os.chdir('..')
        print('\nFinished. Print the summary with vib.summary()\n')        
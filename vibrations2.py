""" 
###Vibrations2###
This class performs the vibrational calculation with the following steps.
1st: Calculating Hessian (H) in cartesian coordinate (x, y, z) using ASE vibrations module
2nd: Refining the H in the normal coordinate (qi)
3rd: Repeat 2nd until the ZPE difference between two consecutive calculation is below error_thr. 

The derivative of forces is approximated using finite difference methods. 

Some part of the code are copied from ASE. 
https://databases.fysik.dtu.dk/ase/ase/vibrations/vibrations.html

Written by
Samuel Eka Putra Payong Masan
Ph.D. student at Morikawa Group
Osaka University
August 2024
Rev. September 2024
######

Input parameter

    #Following inputs are the same as in ASE's vibration
    atoms: Atoms object
        The atoms to work on.

    indices: list of int
        default=None
        List of indices of atoms to vibrate.  Default behavior is
        to vibrate all atoms.

    delta: float
        default=0.01 Å
        Magnitude of displacements in the 1ts calculation. 
        Also used as the maximum displacement in the following calculation. 

    nfree: int
        default=2
        Number of displacements per atom and cartesian coordinate, 2 and 4 are
        supported. Default is 2 which will displace each atom +delta and
        -delta for each cartesian coordinate.

        
    #Following inputs are NOT exist in ASE's vibration
    method: str
        default="plus_minus"
        "plus": displace along +q
        "plus_minus": displace along +q and -q

    FD_method: str
        default="forces"
        "forces": Use finite difference of forces to calculate derivative.
        "energy": Use finite difference of energy to calculate derivative. This is just experimental. Not recommended to use.

    fmax: float
        default=3e-3 Ry/Bohr
        Target maximum force acting on the displaced atoms. 
        I set this to 3 times relaxation threshold in Quantum ESPRESSO.
        This is used to determine the displacement magnitude, see e.q. https://doi.org/10.1103/PhysRevB.110.075409.
        However, it will limit by 10*delta (see above variable) to avoid too big displacement
        caused by soft modes. 

    error_thr: float
        default=100 meV
        Threshold to stop the calculation. 
        It is definied as the absolute difference of ZPE between two consecutive calculation.
        I set the default to a big value so that the calculation will only be performed two times: 1st in cartesian, 2nd in normal coordinate.
        From my trial, 3rd calculation in the normal coordinate will give almost the same value as the 2nd one.
        If you want to do more iteration, reduce the threshold. 

    isolated: bool
        default=False
        Wether the system is an isolated molecule (True) or not (False).
        One can use this to neglect rotational and translational mode,
        they will be printed as zero. 
    
        if isolated=True please also specify following input
            mol_shape: str
                default='None'
                The shape of molecule. It is either 'linear' or 'nonlinear'    

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
    >>> vib = Vibrations2(atoms, isolated=True, shape='linear')
    >>> vib.run()

    ###Step : 1###
    Displacement in cartesian coordinate.
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

    ###Step : 2###
    Displacement in normal coordiate.
    ZPE difference with prev. calc. = 2.76 meV
    ---------------------
    #    meV     cm^-1
    ---------------------
    0    0.0       0.0
    1    0.0       0.0
    2    0.0       0.0
    3    0.0       0.0
    4    0.0       0.0
    5  152.6    1231.3
    ---------------------
    Zero-point energy: 0.081 eV
    >>> vib.write_mode(-1)  # write last mode to trajectory file
    >>> vib.write_disp()  # write the displacement in xyz file. This can be used to restart a calculation with vib.run_from_disp().

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
from copy import copy
from ase.calculators.singlepoint import SinglePointCalculator

class Vibrations2():
    def __init__(self, atoms, 
                 indices=None, 
                 delta=0.01, 
                 nfree=2, 
                 isolated=False, 
                 mol_shape=None, 
                 FD_method='forces', 
                 fmax=3e-3*units.Ry/units.Bohr,
                 error_thr = 100, #Error in meV
                 method = 'plus_minus',                 
                 ):

        self.atoms = atoms
        self.calc = atoms.calc
        if indices == None:
            self.indices = [atom.index for atom in self.atoms]
        else:
            self.indices = indices
        self.delta = delta
        self.nfree = nfree
        
        self.isolated = isolated
        self.mol_shape = mol_shape

        self.error_thr = error_thr
        self.FD_method = FD_method
        self.fmax = fmax
        self.method = method
                     
        if len(self.atoms) == len(self.indices):
            if self.isolated:
                if self.mol_shape is None:
                    raise ValueError("Please set the molecule shape for isolated molecules.")
            else:
                raise ValueError("Isolated system detected. Please set isolated=True and specify the molecule shape.")

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
        The resulting normal modes will be used for 
        further calculation. 
        """
        vib = Vibrations(self.atoms, 
                         indices=self.indices, 
                         delta=self.delta,
                         name = 'vib1', #Do not change this!
                         nfree = self.nfree,
                         )
        vib.run()        
        vib.summary()            
        return vib
        
    def run(self):        
        """
        Main part of the code.
        """       
        step = 1
        self.step = step 
        print(f'\n###Step : {step}###')
        print('Displacement in cartesian coordinate.')
        # Getting the frequencies and normal modes
        vib1 = self.do_vib1()
        energies = vib1.get_energies()
        zpe = self.get_zpe(energies)
        modes = [vib1.get_mode(i) for i in range(len(energies))]
        modes = np.array(modes)

        error = self.error_thr + 1                

        while error > self.error_thr:
            
            step += 1
            self.step = step
            print(f'\n###Step : {step}###')
            print('Displacement in normal coordiate.')
            # Make directory for storing the displacement files
            try:
                os.mkdir(f'vib{step}')    
            except:
                pass
        
            # Calculate the forces of the displaced atoms along normal coordinate qi, then calculate the H(qi,qj)
            
            H_q = np.zeros((3*len(self.indices), 3*len(self.indices)))
            i = 3*len(self.indices) - 1
            self.all_delta = np.zeros(3*len(self.indices))
            for _ in range(self.Nmode):
                factor = self.get_amplitude(energies[i])
                displacement = factor*modes[i]                   

                # Copy the atomic structure and make displacement in +Q direction 
                try:
                    atoms_displaced_plus = read(f'vib{step}/disp{i}_plus.xyz')
                    e_plus = atoms_displaced_plus.get_potential_energy()
                except:                        
                    atoms_displaced_plus = self.atoms.copy()
                    atoms_displaced_plus.positions += displacement
                    atoms_displaced_plus.calc = copy(self.calc)
                    e_plus = atoms_displaced_plus.get_potential_energy()
                    atoms_displaced_plus.write(f'vib{step}/disp{i}_plus.xyz')
                f_plus = atoms_displaced_plus.get_forces()

                if self.method == 'plus_minus':
                    # Copy the atomic structure and make displacement in -Q direction 
                    try:
                        atoms_displaced_min = read(f'vib{step}/disp{i}_min.xyz')
                        e_min = atoms_displaced_min.get_potential_energy()
                    except:    
                        atoms_displaced_min = self.atoms.copy()
                        atoms_displaced_min.positions -= displacement
                        atoms_displaced_min.calc = copy(self.calc)
                        e_min = atoms_displaced_min.get_potential_energy()
                        atoms_displaced_min.write(f'vib{step}/disp{i}_min.xyz')
                    f_min = atoms_displaced_min.get_forces()

                #Take dot product between forces and the q
                u = modes[i]
                f_plus_q = f_plus[self.indices].ravel() @ u[self.indices].ravel().T
                if self.method == 'plus_minus':
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
                self.all_delta[i] = delta

                if self.method == 'plus_minus':
                    if self.FD_method == 'energy':
                        e_eq = self.atoms.get_potential_energy()
                        # Probably better in avoiding eggbox effect?                    
                        H_q[i,i] = (e_min + e_plus - 2 * e_eq)/(delta)**2
                    elif self.FD_method == 'forces':
                        H_q[i,i] =  (f_min_q - f_plus_q)/(2*delta) #- f_plus_q/delta #
                else:
                    if self.FD_method == 'energy':
                        e_eq = self.atoms.get_potential_energy()
                        # Probably better in avoiding eggbox effect?                    
                        H_q[i,i] = 2*(e_plus - e_eq)/(delta)**2
                    elif self.FD_method == 'forces':
                        H_q[i,i] =  - f_plus_q/delta
    
                i -= 1
                
            masses = self.atoms.get_masses()            
            modes = modes / masses[np.newaxis, :, np.newaxis]**-0.5 
            v = [mode[self.indices].ravel() for mode in modes]
            v = np.array(v)
            v_inv = np.linalg.inv(v)
            H = v.T @ H_q @ v_inv.T
            H /= 2
            H += H.T

            masses = masses[self.indices]
            mass_weights = np.repeat(masses**-0.5, 3)        
        
            H = 1/mass_weights * H * 1/mass_weights[:, np.newaxis]
        
            vibrations = VibrationsData.from_2d(self.atoms, H,
                                          indices=self.indices)

            #Result
            energies = vibrations.get_energies()
            self.energies = energies

            modes = vibrations.get_modes(all_atoms=True)
            self.modes = modes

            #Error
            zpe_new = self.get_zpe(self.energies)
            error = self.get_error(zpe, zpe_new) * 1000
            print(f'ZPE difference with prev. calc. = {error:.2f} meV')
            self.summary()
            zpe = zpe_new

    def get_amplitude(self, e_vib):  
        """
        Calculate the amplitude of the displacement along the Q. 
        The amplitude expect to causses the atomic structure to 
        feel maximum forces of fmax.
        """                   
        e_vib = e_vib.real + e_vib.imag
        s = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
        k = (e_vib/s)**2
        amp = self.fmax / k    
        if amp > 10*self.delta:
            amp = 10*self.delta
        return amp
    
    def get_error(self, zpe_old, zpe_new):        
        return abs(zpe_old - zpe_new)
        
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

        with ase.io.Trajectory('%s.%d.traj' % ('mode', n), 'w') as traj:
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
    
    def get_zpe(self, energies):
        return 0.5 * np.asarray(energies).real.sum()
    
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
            atoms_displaced = read(f'vib{self.step}/disp{i}_plus.xyz')            
            pos_disp = atoms_displaced.get_positions()
            disp = pos_disp - pos_eq
            atoms_disp_only = self.atoms.copy()
            atoms_disp_only.set_positions(disp, apply_constraint=False)
            atoms_disp_only.write(f'{dir}/disp{i}.xyz')

            i -= 1
        
        np.savetxt(f'{dir}/all_delta.txt', self.all_delta)

    def run_from_disp(self):
        """
        One can directly calculate the eigen value if 
        the displacement vectors is present inside the disp directory.
        """        
        
        print('\nDisplacement coordinate presupplied.')

        # Make directory for storing the displacement files
        dir = 'vib_from_disp'
        try:
            os.mkdir(dir)    
        except:
            pass
        
        # Calculate the forces of the displaced atoms along normal coordinate qi
            
        H_q = np.zeros((3*len(self.indices), 3*len(self.indices)))
        i = 3*len(self.indices) - 1
        modes = []
        all_delta = np.loadtxt('disp/all_delta.txt')
        for _ in range(self.Nmode):
            displacement = read(f'disp/disp{i}.xyz')
            displacement = displacement.get_positions()
            
            delta = all_delta[i]
            mode = displacement / delta
            modes.append(mode)
            # Copy the atomic structure and make displacement in +Q direction 
            try:
                atoms_displaced_plus = read(f'{dir}/disp{i}_plus.xyz')
                e_plus = atoms_displaced_plus.get_potential_energy()
            except:                        
                atoms_displaced_plus = self.atoms.copy()
                atoms_displaced_plus.positions += displacement
                atoms_displaced_plus.calc = copy(self.calc)
                e_plus = atoms_displaced_plus.get_potential_energy()
                atoms_displaced_plus.write(f'{dir}/disp{i}_plus.xyz')
            f_plus = atoms_displaced_plus.get_forces()

            if self.method == 'plus_minus':
                # Copy the atomic structure and make displacement in -Q direction 
                try:
                    atoms_displaced_min = read(f'{dir}/disp{i}_min.xyz')
                    e_min = atoms_displaced_min.get_potential_energy()
                except:    
                    atoms_displaced_min = self.atoms.copy()
                    atoms_displaced_min.positions -= displacement
                    atoms_displaced_min.calc = copy(self.calc)
                    e_min = atoms_displaced_min.get_potential_energy()
                    atoms_displaced_min.write(f'{dir}/disp{i}_min.xyz')
                f_min = atoms_displaced_min.get_forces()

            #Take dot product between forces and the q
            u = mode#modes[i]
            f_plus_q = f_plus[self.indices].ravel() @ u[self.indices].ravel().T
            if self.method == 'plus_minus':
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

            if self.method == 'plus_minus':
                if self.FD_method == 'energy':
                    # Probably better in avoiding eggbox effect?          
                    e_eq = self.atoms.get_potential_energy()          
                    H_q[i,i] = (e_min + e_plus - 2 * e_eq)/(delta)**2
                elif self.FD_method == 'forces':
                    H_q[i,i] =  (f_min_q - f_plus_q)/(2*delta) #- f_plus_q/delta #
            else:
                if self.FD_method == 'energy':
                    # Probably better in avoiding eggbox effect?                    
                    e_eq = self.atoms.get_potential_energy()
                    H_q[i,i] = 2*(e_plus - e_eq)/(delta)**2
                elif self.FD_method == 'forces':
                    H_q[i,i] =  - f_plus_q/delta
    
            i -= 1
        
        modes.reverse()
        masses = self.atoms.get_masses() 
        modes = modes/ masses[np.newaxis, :, np.newaxis]**-0.5 
        v = [mode[self.indices].ravel() for mode in modes]
        v = np.array(v)
        v_inv = np.linalg.inv(v)
        H = v.T @ H_q @ v_inv.T
        H /= 2
        H += H.T

        masses = masses[self.indices]
        mass_weights = np.repeat(masses**-0.5, 3)        
        
        H = 1/mass_weights * H * 1/mass_weights[:, np.newaxis]
        
        vibrations = VibrationsData.from_2d(self.atoms, H,
                                          indices=self.indices)

        #Result
        energies = vibrations.get_energies()
        self.energies = energies

        modes = vibrations.get_modes(all_atoms=True)
        self.modes = modes

        self.summary()

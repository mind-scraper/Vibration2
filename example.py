from ase.optimize import BFGS as relaxer
from ase import Atoms
from ase.calculators.emt import EMT
from vibrations2 import Vibrations2

atoms = Atoms('N2', [(0, 0, 0), (0, 0, 1.1)], calculator=EMT())

# Optimize the geometry
opt = relaxer(atoms)
opt.run(fmax=0.01)

# Vibrational analysis
vib = Vibrations2(atoms, isolated=True, mol_shape='linear')
vib.run()
vib.summary()
vib.write_mode(-1)
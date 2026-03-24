"""Unit conversion constants used by CC and NMP workflows.

All values are defined in SI-compatible forms and imported where needed.
"""

import scipy.constants as const

# Length conversion: 1 bohr = 0.529177... angstrom.
# Used in `ccdiagram.configurational_coordinate` for phonon-energy evaluation.
BOHR2ANG = const.physical_constants["Bohr radius"][0] / 1e-10

# Energy conversion: 1 Hartree = 27.211386... eV.
# Used in `ccdiagram.configurational_coordinate` for curvature-to-energy conversion.
HARTREE2EV = const.physical_constants["Hartree energy in eV"][0]

# Mass conversion: 1 amu = 1822.888... electron masses.
# Used in `ccdiagram.configurational_coordinate` for modal-mass normalization.
AMU2EMASS = const.physical_constants["atomic mass constant-electron mass ratio"][0]

# Gaussian broadening width in eV used by `main.ccdiagram.delta`.
DEFAULT_SMEARING_EV = 0.050

# Reduced Planck constant in eV·s used by NMP transition prefactors.
HBAR = const.hbar / const.e

# Electron volt to Joule conversion used in NMP scaling factors.
EV2J = const.e

# Atomic mass unit to kilogram conversion used in NMP coordinate scaling.
AMU2KG = const.physical_constants["atomic mass constant"][0]

# Angstrom to meter conversion used in NMP coordinate scaling.
ANGS2M = 1e-10

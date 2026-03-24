"""Entry point for the CC-diagram and optional NMP workflow.

Single-run flow:
1) Generate interpolated structures (`--mode generate` or `--mode generate-neb`)
2) Create/submit calculation folders (`generate_system`, `qsub_system`)
3) Fit CC curves and export plots/data (`--mode cc`)
4) Optionally compute NMP capture-time curve (`--run-nmp`)
"""

import argparse

import numpy as np
from mpmath import coth, pi
from scipy.special import factorial

from NanoCore import s2

from cc import (
    configurational_coordinate,
    generate_struct,
    generate_struct_neb,
    generate_system,
    qsub_system,
)
from nmp import compute_capture_time_curve, save_capture_time_curve
from unit import AMU2EMASS, BOHR2ANG, DEFAULT_SMEARING_EV, HARTREE2EV


class ccdiagram(object):
    """Container for CC diagram inputs, constants, and derived properties.

    Parameters
    ----------
    ground : str
        Path to a ground-state SIESTA `.fdf` structure file.
    excited : str
        Path to an excited-state SIESTA `.fdf` structure file.

    Notes
    -----
    Stores conversion constants used during CC fitting:
    - `bohr2ang` (bohr -> Å)
    - `hartree2eV` (Hartree -> eV)
    - `amu2emass` (amu -> electron-mass ratio)
    - `smearing` (Gaussian width in eV)
    """

    def __init__(self, ground, excited):
        self.tol = 1e-5
        self.bohr2ang = BOHR2ANG
        self.hartree2eV = HARTREE2EV
        self.amu2emass = AMU2EMASS
        self.kb = 8.617 * 10 ** (-5)
        self.smearing = DEFAULT_SMEARING_EV
        self.e = 2.71828
        self.gap = 1.516429
        self.npt = 5

        ground_struct = s2.read_fdf(ground)
        excited_struct = s2.read_fdf(excited)

        ground_atoms = ground_struct._atoms
        excited_atoms = excited_struct._atoms

        self.ground_struct = ground_struct
        self.ground_cell = ground_struct._cell
        self.ground_position = np.array([x._position for x in ground_atoms], dtype=float)
        self.ground_mass = np.array([m._mass for m in ground_atoms], dtype=float)

        self.excited_struct = excited_struct
        self.excited_cell = excited_struct._cell
        self.excited_position = np.array([x._position for x in excited_atoms], dtype=float)
        self.excited_mass = np.array([m._mass for m in excited_atoms], dtype=float)

        self.init_info()

    def init_info(self):
        """Print initial geometric distortion diagnostics.

        Side Effects
        ------------
        Writes information to stdout.
        """
        dr, dr2, dR = self.deltaR(self.ground_position, self.excited_position)
        dQ = self.deltaQ(dr2, self.ground_mass)
        print("Calculation configuration coordinate diagram! \n")
        print("Total distortion:                %7.4f" % dR)
        print("Total mass weight distortion:    %7.4f" % dQ)

    def Polynomial2(self, parameters, x):
        """Evaluate a 2nd-order polynomial A*x^2 + B*x + C."""
        A, B, C = parameters
        return A * x ** 2 + B * x + C

    def deltaR(self, pos1, pos2):
        """Compute Cartesian displacement metrics between two structures.

        Parameters
        ----------
        pos1, pos2 : np.ndarray
            Atomic coordinates in Å with shape (N, 3).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, float]
            `(dr, dr2, dR)` where `dr` is per-atom displacement vector in Å,
            `dr2` is squared norm per atom in Å^2, and `dR` is total distortion in Å.

        Side Effects
        ------------
        Prints per-atom displacements and totals to stdout.
        """
        dr = pos2 - pos1
        dr2 = np.array([sum(r ** 2) for r in dr], dtype=float)
        print(sum(dr ** 2))
        dR = np.sqrt(dr2.sum())

        for i in range(len(dr)):
            print(i + 1)
            print(dr[i])

        return dr, dr2, dR

    def deltaCell(self, vector1, vector2):
        """Return cell-vector difference in Å."""
        return vector2 - vector1

    def deltaQ(self, dr2, mass):
        """Compute mass-weighted distortion amplitude.

        Parameters
        ----------
        dr2 : np.ndarray
            Per-atom squared displacement in Å^2.
        mass : np.ndarray
            Per-atom mass in amu.

        Returns
        -------
        float
            Mass-weighted distortion `dQ` in amu^1/2·Å.
        """
        dq2 = [m * d for m, d in zip(dr2, mass)]
        return np.sqrt(sum(dq2))

    def modalM(self, dQ, dR):
        """Compute modal mass in amu from `dQ` (amu^1/2·Å) and `dR` (Å)."""
        return (dQ / dR) ** 2

    def delta(self, x):
        """Gaussian broadening kernel using `self.smearing` in eV."""
        return np.exp(-(x / self.smearing) ** 2) / (self.smearing * np.sqrt(pi))

    def full_width_half_maxium(self, T, hwg, hwe, Sg, Se):
        """Estimate FWHM of the phonon-broadened optical line.

        Parameters
        ----------
        T : float
            Temperature in K.
        hwg, hwe : float
            Effective phonon energies in meV (overridden by fitted values).
        Sg, Se : float
            Huang-Rhys factors (overridden by fitted values).

        Returns
        -------
        float
            FWHM estimate in eV.
        """
        hwg = self._hwg / 1000
        hwe = self._hwe / 1000
        Sg = self._Sg
        Se = self._Se

        if T == 0:
            WT = np.sqrt(8 * np.log(2)) * Se * hwg / np.sqrt(Sg)
        else:
            WT = (
                np.sqrt(8 * np.log(2))
                * Se
                * hwg
                / np.sqrt(Sg)
                * np.sqrt(coth(hwe / (2 * self.kb * T)))
            )

        return WT

    def dipole_transition(self, n, S):
        """Return Franck-Condon transition weight for vibrational index `n`."""
        return self.e ** (-S) * S ** n / factorial(n, exact=False)


def build_parser():
    """Build CLI parser for CC workflow and optional NMP post-processing."""
    parser = argparse.ArgumentParser(description="CC diagram workflow runner")
    parser.add_argument("ground", help="Ground-state fdf file")
    parser.add_argument("excited", help="Excited-state fdf file")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["generate", "generate-neb", "qsub", "cc"],
        help="Workflow mode",
    )
    parser.add_argument("--run-nmp", action="store_true", help="Run NMP capture-time calculation after cc mode")
    parser.add_argument("--nmp-output", default="nmp.txt", help="Output file path for NMP capture-time results")
    parser.add_argument("--nmp-tmin", type=float, default=5.0, help="Minimum temperature for NMP curve")
    parser.add_argument("--nmp-tmax", type=float, default=2000.0, help="Maximum temperature for NMP curve")
    parser.add_argument("--nmp-tnum", type=int, default=400, help="Number of temperature points for NMP curve")
    return parser


def main():
    """Execute the selected CC workflow mode.

    Side Effects
    ------------
    Depending on mode, creates directories/files, may submit jobs via `sbatch`,
    and writes CC/NMP output artifacts.
    """
    args = build_parser().parse_args()
    diagram = ccdiagram(args.ground, args.excited)

    if args.mode == "generate":
        generate_struct(diagram)
        generate_system(diagram)
    elif args.mode == "generate-neb":
        generate_struct_neb(diagram)
        generate_system(diagram)
    elif args.mode == "qsub":
        qsub_system(diagram)
    elif args.mode == "cc":
        configurational_coordinate(diagram)
        if args.run_nmp:
            temperatures = np.linspace(args.nmp_tmin, args.nmp_tmax, args.nmp_tnum)
            wi = diagram._hwg / 1000.0
            wf = diagram._hwe / 1000.0
            T_array, capture_time_array = compute_capture_time_curve(
                dQ=diagram._dQ,
                dE=diagram._dE,
                wi=wi,
                wf=wf,
                temperatures=temperatures,
            )
            save_capture_time_curve(args.nmp_output, T_array, capture_time_array)


if __name__ == "__main__":
    main()

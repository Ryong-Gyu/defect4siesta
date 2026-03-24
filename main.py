import argparse

import numpy as np
from mpmath import coth, pi
from scipy.special import factorial

from NanoCore import s2

from ccdiagram_workflow import (
    configurational_coordinate,
    generate_struct,
    generate_struct_neb,
    generate_system,
    qsub_system,
)


class ccdiagram(object):
    def __init__(self, ground, excited):
        self.tol = 1e-5
        self.bohr2ang = 0.52918
        self.hartree2eV = 27.2114
        self.amu2emass = 1822.89
        self.kb = 8.617 * 10 ** (-5)
        self.smearing = 0.050
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
        dr, dr2, dR = self.deltaR(self.ground_position, self.excited_position)
        dQ = self.deltaQ(dr2, self.ground_mass)
        print("Calculation configuration coordinate diagram! \n")
        print("Total distortion:                %7.4f" % dR)
        print("Total mass weight distortion:    %7.4f" % dQ)

    def Polynomial2(self, parameters, x):
        A, B, C = parameters
        return A * x ** 2 + B * x + C

    def deltaR(self, pos1, pos2):
        dr = pos2 - pos1
        dr2 = np.array([sum(r ** 2) for r in dr], dtype=float)
        print(sum(dr ** 2))
        dR = np.sqrt(dr2.sum())

        for i in range(len(dr)):
            print(i + 1)
            print(dr[i])

        return dr, dr2, dR

    def deltaCell(self, vector1, vector2):
        return vector2 - vector1

    def deltaQ(self, dr2, mass):
        dq2 = [m * d for m, d in zip(dr2, mass)]
        return np.sqrt(sum(dq2))

    def modalM(self, dQ, dR):
        return (dQ / dR) ** 2

    def delta(self, x):
        return np.exp(-(x / self.smearing) ** 2) / (self.smearing * np.sqrt(pi))

    def full_width_half_maxium(self, T, hwg, hwe, Sg, Se):
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
        return self.e ** (-S) * S ** n / factorial(n, exact=False)


def build_parser():
    parser = argparse.ArgumentParser(description="CC diagram workflow runner")
    parser.add_argument("ground", help="Ground-state fdf file")
    parser.add_argument("excited", help="Excited-state fdf file")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["generate", "generate-neb", "cc"],
        help="Workflow mode",
    )
    return parser


def main():
    args = build_parser().parse_args()
    diagram = ccdiagram(args.ground, args.excited)

    if args.mode == "generate":
        generate_struct(diagram)
        generate_system(diagram)
        qsub_system(diagram)
    elif args.mode == "generate-neb":
        generate_struct_neb(diagram)
        generate_system(diagram)
        qsub_system(diagram)
    elif args.mode == "cc":
        configurational_coordinate(diagram)


if __name__ == "__main__":
    main()

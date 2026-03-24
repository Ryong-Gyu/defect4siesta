import copy
import glob
import os

import numpy as np
from scipy.optimize import fminbound, fsolve
from NanoCore import s2, Vector
from plotting import save_ccdiagram_plot


def _write_info(diagram):
    """Print and export fitted CC parameters.

    Side Effects
    ------------
    Writes `Effective_parameters.dat`, `CCdiagram.dat`, `Ground.dat`,
    `Excited.dat`, and `Conduction.dat` in the current directory.
    """
    print("Total distortion               dR:  %7.4f [ang]" % diagram._dR)
    print("Total mass weight distortion   dQ:  %7.4f [amu^1/2*ang]" % diagram._dQ)
    print("Modal mass                      M:  %7.4f [amu]" % diagram._M)
    print("Effective phonon energy (g)   hwg:  %7.4f [meV]" % diagram._hwg)
    print("Effective phonon energy (e)   hwg:  %7.4f [meV]" % diagram._hwe)
    print("Zero phonon line             Ezpl:  %7.4f [eV]" % diagram._ZPL)
    print("Activation energy            Eact:  %7.4f [eV]" % diagram._Eact)
    print("Activation energy (conduction)   :  %7.4f [eV]" % diagram._dE)
    print("Absorption energy             Eabs:  %7.4f [eV]" % diagram._Eabs)
    print("Emission energy             Eems:  %7.4f [eV]" % diagram._Eems)
    print("Huang-Rhys factor (g)           Sg:  %7.4f" % diagram._Sg)
    print("Huang-Rhys factor (e)           Se:  %7.4f" % diagram._Se)
    print("Franck-Condon shift (g)           dfc_g:  %7.4f" % diagram._dfc_g)
    print("Franck-Condon shift (e)           dfc_e:  %7.4f" % diagram._dfc_e)

    with open("Effective_parameters.dat", "w") as file:
        file.write("Total distortion               dR:  %7.4f [ang]\n" % diagram._dR)
        file.write("Total mass weight distortion   dQ:  %7.4f [amu^1/2*ang]\n" % diagram._dQ)
        file.write("Modal mass                      M:  %7.4f [amu]\n" % diagram._M)
        file.write("Effective phonon energy (g)   hwg:  %7.4f [meV]\n" % diagram._hwg)
        file.write("Effective phonon energy (e)   hwg:  %7.4f [meV]\n" % diagram._hwe)
        file.write("Zero phonon line             Ezpl:  %7.4f [eV]\n" % diagram._ZPL)
        file.write("Activation energy            Eact:  %7.4f [eV]\n" % diagram._Eact)
        file.write("Activation energy (conduction)   :  %7.4f [eV]\n" % diagram._dE)
        file.write("Absorption energy             Eabs:  %7.4f [eV]\n" % diagram._Eabs)
        file.write("Emission energy             Eems:  %7.4f [eV]\n" % diagram._Eems)
        file.write("Huang-Rhys factor (g)           Sg:  %7.4f\n" % diagram._Sg)
        file.write("Huang-Rhys factor (e)           Se:  %7.4f\n" % diagram._Se)
        file.write("Franck-Condon shift (g)           dfc_g:  %7.4f\n" % diagram._dfc_g)
        file.write("Franck-Condon shift (e)           dfc_e:  %7.4f\n" % diagram._dfc_e)

    with open("CCdiagram.dat", "w") as file2:
        n_q = len(diagram._plotQ)
        for i in range(n_q):
            file2.write(
                "%10.4f %10.4f %10.4f %10.4f\n"
                % (diagram._plotQ[i], diagram._plotEg[i], diagram._plotEe[i], diagram._plotEc[i])
            )

    with open("Ground.dat", "w") as file3:
        for q, e in zip(diagram._plotQ1, diagram._plotE1):
            file3.write("%10.4f %10.4f\n" % (q, e))

    with open("Excited.dat", "w") as file4:
        for q, e in zip(diagram._plotQ2, diagram._plotE2):
            file4.write("%10.4f %10.4f\n" % (q, e))

    with open("Conduction.dat", "w") as file5:
        for q, e in zip(diagram._plotQ3, diagram._plotE3):
            file5.write("%10.4f %10.4f\n" % (q, e))


def generate_struct(diagram):
    """Generate linearly interpolated structures at fixed cell.

    Side Effects
    ------------
    Writes `linear*` structure files in the current directory.
    """
    init_struct = diagram.ground_struct
    dr, _, _ = diagram.deltaR(diagram.ground_position, diagram.excited_position)
    ddr = dr / diagram.npt

    for iddr in range(-2 * diagram.npt, 2 * diagram.npt):
        struct = copy.copy(init_struct)
        for iatom, delta in enumerate(ddr):
            pos = diagram.ground_position[iatom]
            pos2 = Vector(pos + iddr * delta)
            struct._atoms[iatom].set_position(pos2)
        s2.Siesta(struct).write_struct()
        os.system("mv STRUCT.fdf linear%02d" % iddr)


def generate_struct_neb(diagram):
    """Generate linearly interpolated structures including cell interpolation.

    Side Effects
    ------------
    Writes `linear*` structure files in the current directory.
    """
    init_struct = diagram.ground_struct

    dr, _, _ = diagram.deltaR(diagram.ground_position, diagram.excited_position)
    d_cell = diagram.deltaCell(diagram.ground_cell, diagram.excited_cell)

    ddr = dr / diagram.npt
    dd_cell = d_cell / diagram.npt

    for iddr in range(-2 * diagram.npt, 2 * diagram.npt):
        struct = copy.copy(init_struct)
        for iatom, delta in enumerate(ddr):
            pos = diagram.ground_position[iatom]
            pos2 = Vector(pos + iddr * delta)
            struct._atoms[iatom].set_position(pos2)
        struct._cell = diagram.ground_cell + iddr * dd_cell
        s2.Siesta(struct).write_struct()
        os.system("mv STRUCT.fdf linear%02d" % iddr)


def generate_system(diagram):
    """Create calculation directories and copy inputs for each interpolated structure.

    Side Effects
    ------------
    Creates `*_calc` directories and many subdirectories/files. Changes the
    working directory temporarily during setup.
    """
    os.system("mkdir ground_calc")
    os.system("mkdir excited_calc")
    os.system("mkdir conduction_calc")

    struct_files = glob.glob("linear*")

    for state in ["ground", "excited", "conduction"]:
        os.chdir(state + "_calc")
        for istruct in struct_files:
            os.system("mkdir %s" % istruct)
            os.system("cp -r ../%s/input/ %s/." % (state, istruct))
            os.system("cp ../%s %s/input/STRUCT.fdf" % (istruct, istruct))
            os.system("cp ../%s/slm* %s/." % (state, istruct))
        os.chdir("..")


def qsub_system(diagram):
    """Submit each prepared structure calculation via `sbatch`.

    Side Effects
    ------------
    Changes directories while traversing structure folders and launches jobs.
    """
    struct_files = glob.glob("linear*")

    for state in ["ground", "excited", "conduction"]:
        os.chdir(state + "_calc")
        for istruct in struct_files:
            os.chdir("%s" % istruct)
            if not os.path.isdir("OUT"):
                os.system("sbatch slm*")
            os.chdir("..")
        os.chdir("..")


def get_total_energy(diagram):
    """Collect converged total energies for each charge/state branch.

    Returns
    -------
    tuple[np.ndarray, ...]
        Branch Q grids and corresponding energies in eV.

    Side Effects
    ------------
    Traverses calculation directories via `os.chdir` and prints progress.
    """
    struct_files = sorted(glob.glob("linear*"))

    ground_energy, excited_energy, conduction_energy = [], [], []
    q1, q2, q3 = [], [], []

    for state in ["ground", "excited", "conduction"]:
        print(state)
        os.chdir(state + "_calc")
        for istruct in struct_files:
            if not os.path.isdir(istruct):
                continue

            os.chdir("%s" % istruct)
            if os.path.isdir("OUT"):
                os.chdir("OUT")

                try:
                    e = s2.get_total_energy()
                except IndexError:
                    print("not converged")
                else:
                    qtmp = float(istruct.split("linear")[-1]) / diagram.npt
                    # Fit windows are branch-specific to isolate locally harmonic
                    # regions around each minimum/crossing point.
                    if state == "ground" and (-1 - diagram.tol <= qtmp <= 1 + diagram.tol):
                        ground_energy.append(e)
                        q1.append(qtmp)
                    elif state == "conduction" and (-0.9 - diagram.tol <= qtmp <= 0 + diagram.tol):
                        conduction_energy.append(e)
                        q3.append(qtmp)
                    elif state == "excited" and (0 - diagram.tol <= qtmp <= 2 + diagram.tol):
                        excited_energy.append(e)
                        q2.append(qtmp)

                os.chdir("..")
            os.chdir("..")
        os.chdir("..")

    q1 = np.array(q1, dtype=float)
    q2 = np.array(q2, dtype=float)
    q3 = np.array(q3, dtype=float)

    print("Number of calculated data (ground): %d" % len(q1))
    print("Number of calculated data (excited): %d" % len(q2))

    return (
        q1,
        q2,
        q3,
        np.array(ground_energy, dtype=float),
        np.array(excited_energy, dtype=float),
        np.array(conduction_energy, dtype=float),
    )


def configurational_coordinate(diagram):
    """Fit CC parabolas and derive effective parameters.

    Returns
    -------
    None
        Results are stored on `diagram` fields (`_dQ`, `_Eact`, `_hwg`, ...).

    Side Effects
    ------------
    Calls plotting and writes multiple `.dat` output files in cwd.
    """
    q1, q2, q3, ground_energy, excited_energy, conduction_energy = get_total_energy(diagram)

    parameter1 = np.polyfit(q1, ground_energy, 2)
    parameter2 = np.polyfit(q2, excited_energy, 2)
    parameter3 = np.polyfit(q3, conduction_energy, 2)

    ground_function = lambda x: diagram.Polynomial2(parameter1, x)
    excited_function = lambda x: diagram.Polynomial2(parameter2, x)
    conduction_function = lambda x: diagram.Polynomial2(parameter3, x)

    qg = fminbound(ground_function, min(q1), max(q1))
    qe = fminbound(excited_function, min(q2), max(q2))
    qc = fminbound(conduction_function, min(q3), max(q3))

    dr, dr2, dR = diagram.deltaR(diagram.ground_position, diagram.excited_position)
    dQ = diagram.deltaQ(dr2, diagram.ground_mass)
    modal_mass = diagram.modalM(dQ, dR)

    q_cross = fsolve(lambda x: ground_function(x) - excited_function(x), 1)
    q_conduction = fsolve(lambda x: conduction_function(x) - excited_function(x), 0)

    dE = excited_function(q_conduction) - conduction_function(qc)
    dEg = ground_function(qe) - ground_function(qg)
    dEe = excited_function(qg) - excited_function(qe)

    effective_phonon_energy_g = np.sqrt((2 * dEg / diagram.hartree2eV) / (dQ**2 / diagram.bohr2ang**2 * diagram.amu2emass)) * diagram.hartree2eV
    effective_phonon_energy_e = np.sqrt((2 * dEe / diagram.hartree2eV) / (dQ**2 / diagram.bohr2ang**2 * diagram.amu2emass)) * diagram.hartree2eV

    dhw = effective_phonon_energy_e - effective_phonon_energy_g
    activation_energy = excited_function(q_cross) - excited_function(qe)
    absorption_energy = excited_function(qg) - ground_function(qg)
    emission_energy = excited_function(qe) - ground_function(qe)
    zero_phonon_line = excited_function(qe) - ground_function(qg) + dhw
    binding_energy = conduction_function(qc) - excited_function(qe)

    dfc_g = ground_function(qe) - ground_function(qg)
    dfc_e = excited_function(qg) - excited_function(qe)

    huang_rhys_factor_g = dEg / effective_phonon_energy_g
    huang_rhys_factor_e = dEe / effective_phonon_energy_e

    omega_ground = np.sqrt(2 * parameter1[0] / modal_mass)
    omega_excited = np.sqrt(2 * parameter2[0] / modal_mass)

    print("potential barrier: %f \n" % (max(conduction_energy) - min(conduction_energy)))
    print("Binding energy: %f \n" % (binding_energy))

    plot_data = save_ccdiagram_plot(
        dQ,
        parameter1,
        parameter2,
        parameter3,
        q1,
        q2,
        q3,
        ground_energy,
        excited_energy,
        conduction_energy,
    )

    diagram._dR = dR
    diagram._dQ = dQ
    diagram._M = modal_mass
    diagram._hwg = 1000 * effective_phonon_energy_g
    diagram._hwe = 1000 * effective_phonon_energy_e
    diagram._wg = omega_ground
    diagram._we = omega_excited
    diagram._Sg = huang_rhys_factor_g
    diagram._Se = huang_rhys_factor_e
    diagram._dEg = dEg
    diagram._dEe = dEe
    diagram._ZPL = zero_phonon_line
    diagram._Eabs = absorption_energy
    diagram._Eems = emission_energy
    diagram._Eact = activation_energy
    diagram._dfc_g = dfc_g
    diagram._dfc_e = dfc_e
    diagram._dE = dE

    diagram._plotQ = plot_data["plotQ"]
    diagram._plotEg = plot_data["plotEg"]
    diagram._plotEe = plot_data["plotEe"]
    diagram._plotEc = plot_data["plotEc"]
    diagram._plotQ1 = plot_data["plotQ1"]
    diagram._plotQ2 = plot_data["plotQ2"]
    diagram._plotQ3 = plot_data["plotQ3"]
    diagram._plotE1 = plot_data["plotE1"]
    diagram._plotE2 = plot_data["plotE2"]
    diagram._plotE3 = plot_data["plotE3"]

    _write_info(diagram)

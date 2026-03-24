import numpy as np
import matplotlib.pylab as plt


def save_ccdiagram_plot(
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
):
    fig = plt.figure(figsize=(4, 6))
    ax = fig.add_subplot(111)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    qfit = np.linspace(-20, 20, 300)
    ground_e = np.polyval(parameter1, qfit)
    excited_e = np.polyval(parameter2, qfit)
    conduction_e = np.polyval(parameter3, qfit)

    vbm = min(ground_energy)

    size = 50
    plt.plot(qfit * dQ, ground_e - vbm, color="k", linewidth=3.0, zorder=1)
    plt.plot(qfit * dQ, excited_e - vbm, color="b", linewidth=3.0, zorder=2)

    plt.scatter(q1 * dQ, ground_energy - vbm, s=size, c="w", edgecolors="k", zorder=4, linewidths=2)
    plt.scatter(q2 * dQ, excited_energy - vbm, s=size, c="w", edgecolors="b", zorder=5, linewidths=2)
    plt.scatter(q3 * dQ, conduction_energy - vbm, s=size, c="w", edgecolors="r", zorder=6, linewidths=2)

    plt.ylim(-1, 5)
    plt.xlim(-10, 15)
    plt.xlabel(r"Q [amu$^{1/2} \AA$]", fontsize=18)
    plt.ylabel("Total energy [eV]", fontsize=18)
    plt.tight_layout()
    plt.savefig("ccdiagram.png")

    return {
        "plotQ": qfit * dQ,
        "plotEg": ground_e - vbm,
        "plotEe": excited_e - vbm,
        "plotEc": conduction_e - vbm,
        "plotQ1": q1 * dQ,
        "plotQ2": q2 * dQ,
        "plotQ3": q3 * dQ,
        "plotE1": ground_energy - vbm,
        "plotE2": excited_energy - vbm,
        "plotE3": conduction_energy - vbm,
        "vbm": vbm,
    }

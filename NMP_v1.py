import argparse
from typing import Optional, Union

import numpy as np
from scipy import constants as const
from scipy.interpolate import PchipInterpolator, interp1d

from unit import ANGS2M, AMU2KG, HBAR, EV2J

try:
    from numba import njit, vectorize

    @vectorize
    def herm_vec(x: float, n: int) -> float:
        """Wrap herm function."""
        return herm(x, n)
except ModuleNotFoundError:
    from numpy.polynomial.hermite import hermval

    def njit(*args, **kwargs):      # pylint: disable=W0613
        """Fake njit when numba can't be found."""

        def _njit(func):
            return func

        return _njit

    def herm_vec(x: float, n: int) -> float:
        """Wrap hermval function."""
        return hermval(x, [0.0] * n + [1.0])


# Unit conversion parameters
factor = ANGS2M**2 * AMU2KG / HBAR / HBAR / EV2J
Factor3 = 1 / HBAR


@njit(cache=True)
def fact(n: int) -> float:
    """Compute the factorial of n."""
    lookup_table = np.array(
        [
            1,
            1,
            2,
            6,
            24,
            120,
            720,
            5040,
            40320,
            362880,
            3628800,
            39916800,
            479001600,
            6227020800,
            87178291200,
            1307674368000,
            20922789888000,
            355687428096000,
            6402373705728000,
            121645100408832000,
            2432902008176640000,
        ],
        dtype=np.double,
    )

    if n > 20:
        return lookup_table[-1] * np.prod(np.array(list(range(21, n + 1)), dtype=np.double))
    return lookup_table[n]


@njit(cache=True)
def herm(x: float, n: int) -> float:
    """Recursive definition of hermite polynomial."""
    if n == 0:
        return 1.0
    if n == 1:
        return 2.0 * x

    y1 = 2.0 * x
    dy1 = 2.0
    for i in range(2, n + 1):
        yn = 2.0 * x * y1 - dy1
        dyn = 2.0 * i * y1
        y1 = yn
        dy1 = dyn
    return yn


@njit(cache=True)
def overlap_NM(
    DQ: float,
    w1: float,
    w2: float,
    n1: int,
    n2: int,
    qq: np.ndarray,
) -> float:
    """Compute overlap between two displaced harmonic oscillators."""
    hn1_q = herm_vec(np.sqrt(factor * w1) * (qq - DQ), n1)
    hn2_q = herm_vec(np.sqrt(factor * w2) * qq, n2)

    wfn1 = (factor * w1 / np.pi) ** (0.25) * (1.0 / np.sqrt(2.0**n1 * fact(n1))) * hn1_q * np.exp(
        -(factor * w1) * (qq - DQ) ** 2 / 2.0
    )
    wfn2 = (factor * w2 / np.pi) ** (0.25) * (1.0 / np.sqrt(2.0**n2 * fact(n2))) * hn2_q * np.exp(
        -(factor * w2) * qq**2 / 2.0
    )

    return np.trapz(wfn2 * wfn1, x=qq)


def build_qq_grid(qq_min: float = -30.0, qq_max: float = 30.0, qq_points: int = 5000) -> np.ndarray:
    """Build integration grid for overlap calculations."""
    return np.linspace(qq_min, qq_max, qq_points)


def get_C(
    dQ: float,
    dE: float,
    wi: float,
    wf: float,
    Wif: float,
    volume: float,
    g: int = 1,
    T: Union[float, np.ndarray] = 300.0,
    sigma: Union[str, float] = "pchip",
    occ_tol: float = 1e-5,
    qq: Optional[np.ndarray] = None,
) -> Union[float, np.ndarray]:
    """Compute the nonradiative capture coefficient."""
    del Wif, volume, g  # kept for API compatibility

    if qq is None:
        qq = build_qq_grid()

    kT = (const.k / const.e) * T
    Z = 1.0 / (1 - np.exp(-wi / kT))

    Ni, Nf = (17, 50)
    tNi = np.ceil(-np.max(kT) * np.log(occ_tol) / wi).astype(int)
    if tNi > Ni:
        Ni = tNi
    tNf = np.ceil((dE + Ni * wi) / wf).astype(int)
    if tNf > Nf:
        Nf = tNf

    ovl = np.zeros((Ni, Nf), dtype=np.longdouble)
    for m in np.arange(Ni):
        for n in np.arange(Nf):
            ovl[m, n] = overlap_NM(dQ, wi, wf, m, n, qq)

    t = np.linspace(-Ni * wi, Nf * wf, 5000)
    R = 0.0
    for m in np.arange(Ni - 1):
        weight_m = np.exp(-m * wi / kT) / Z
        if isinstance(sigma, str):
            E, matels = (np.zeros(Nf), np.zeros(Nf))
            for n in np.arange(Nf):
                matel = np.sqrt(Factor3) * dQ * ovl[m, n]
                E[n] = n * wf - m * wi
                matels[n] = np.abs(np.conj(matel) * matel)
            if sigma[0].lower() == "c":
                f = interp1d(E, matels, kind="cubic", bounds_error=False, fill_value=0.0)
            else:
                f = PchipInterpolator(E, matels, extrapolate=False)
            R = R + weight_m * (f(dE) * np.sum(matels) / np.trapz(np.nan_to_num(f(t)), x=t))
        else:
            for n in np.arange(Nf):
                delta = np.exp((dE + n * wf - m * wi) / (2.0 * sigma**2)) / (sigma * np.sqrt(2.0 * np.pi))
                matel = np.sqrt(Factor3) * ovl[m, n]
                R = R + weight_m * delta * np.abs(np.conj(matel) * matel)

    return R


def compute_capture_time_curve(
    dQ: float,
    dE: float,
    wi: float,
    wf: float,
    temperatures: np.ndarray,
    Wif: float = 1.0,
    volume: float = 1.0,
    g: int = 1,
    sigma: Union[str, float] = "pchip",
    occ_tol: float = 1e-10,
    qq: Optional[np.ndarray] = None,
):
    """Return temperature array and capture-time curve."""
    if qq is None:
        qq = build_qq_grid()

    capture_coefficients = np.array(
        [get_C(dQ, dE, wi, wf, Wif, volume, g, temp, sigma, occ_tol, qq=qq) for temp in temperatures]
    )
    capture_time = 1 / capture_coefficients
    return temperatures, capture_time


def save_capture_time_curve(
    output_path: str,
    temperatures: np.ndarray,
    capture_time: np.ndarray,
):
    """Persist capture-time curve in legacy nmp.txt format."""
    with open(output_path, "w", encoding="utf-8") as file:
        for temp, ctime in zip(temperatures, capture_time):
            file.write(f"{temp:.0f}  {ctime}\n")


def _build_parser():
    parser = argparse.ArgumentParser(description="NMP capture-time calculator")
    parser.add_argument("--dQ", type=float, required=True)
    parser.add_argument("--dE", type=float, required=True)
    parser.add_argument("--wi", type=float, required=True)
    parser.add_argument("--wf", type=float, required=True)
    parser.add_argument("--output", default="nmp.txt")
    parser.add_argument("--tmin", type=float, default=5.0)
    parser.add_argument("--tmax", type=float, default=2000.0)
    parser.add_argument("--tnum", type=int, default=400)
    return parser


def main():
    args = _build_parser().parse_args()
    temperatures = np.linspace(args.tmin, args.tmax, args.tnum)
    T_array, capture_time_array = compute_capture_time_curve(
        dQ=args.dQ,
        dE=args.dE,
        wi=args.wi,
        wf=args.wf,
        temperatures=temperatures,
    )
    save_capture_time_curve(args.output, T_array, capture_time_array)


if __name__ == "__main__":
    main()

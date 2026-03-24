import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as const
import warnings
from typing import Union

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
        return hermval(x, [0.]*n + [1.])


# Unit conversion parameters
factor = ANGS2M**2 * AMU2KG / HBAR / HBAR / EV2J   ## m/hbar
Factor2 = const.hbar / ANGS2M**2 / AMU2KG
Factor3 = 1 / HBAR
k = 1.38*1e-23

@njit(cache=True)
def fact(n: int) -> float:

    # for fast factorial calculations
    LOOKUP_TABLE = np.array([
        1, 1, 2, 6, 24, 120, 720, 5040, 40320,
        362880, 3628800, 39916800, 479001600,
        6227020800, 87178291200, 1307674368000,
        20922789888000, 355687428096000, 6402373705728000,
        121645100408832000, 2432902008176640000], dtype=np.double)

    """Compute the factorial of n."""
    if n > 20:
        return LOOKUP_TABLE[-1] * \
            np.prod(np.array(list(range(21, n+1)), dtype=np.double))
    return LOOKUP_TABLE[n]


@njit(cache=True)
def herm(x: float, n: int) -> float:
    """Recursive definition of hermite polynomial."""
    if n == 0:
        return 1.
    if n == 1:
        return 2. * x

    y1 = 2. * x
    dy1 = 2.
    for i in range(2, n+1):
        yn = 2. * x * y1 - dy1
        dyn = 2. * i * y1
        y1 = yn
        dy1 = dyn
    return yn


@njit(cache=True)
def overlap_NM(
        DQ: float,
        w1: float,
        w2: float,
        n1: int,
        n2: int
) -> float:
    """Compute the overlap between two displaced harmonic oscillators.

    This function computes the overlap integral between two harmonic
    oscillators with frequencies w1, w2 that are displaced by DQ for the
    quantum numbers n1, n2. The integral is computed using the trapezoid
    method and the analytic form for the wavefunctions.

    Parameters
    ----------
    DQ : float
        displacement between harmonic oscillators in amu^{1/2} Angstrom
    w1, w2 : float
        frequencies of the harmonic oscillators in eV
    n1, n2 : integer
        quantum number of the overlap integral to calculate

    Returns
    -------
    np.longdouble
        overlap of the two harmonic oscillator wavefunctions
    """
    Hn1Q = herm_vec(np.sqrt(factor*w1)*(QQ-DQ), n1)   ## 하모닉 포텐셜1 에르미트 다항식 --> 축이 DQ만큼 이동된 모습 
    Hn2Q = herm_vec(np.sqrt(factor*w2)*(QQ), n2)      ## 하모닉 포텐셜2 에르미트 다항식 --> 축이 0으로 고정 

    wfn1 = (factor*w1/np.pi)**(0.25)*(1./np.sqrt(2.**n1*fact(n1))) * \
        Hn1Q*np.exp(-(factor*w1)*(QQ-DQ)**2/2.)     ## 파동함수1 
    wfn2 = (factor*w2/np.pi)**(0.25)*(1./np.sqrt(2.**n2*fact(n2))) * \
        Hn2Q*np.exp(-(factor*w2)*QQ**2/2.)          ## 파동함수2

    return np.trapz(wfn2*wfn1, x=QQ)               


def get_C(
        dQ: float,
        dE: float,
        wi: float,
        wf: float,
        Wif: float,
        volume: float,
        g: int = 1,
        T: Union[float, np.ndarray] = 300.,
        sigma: Union[str, float] = 'pchip',
        occ_tol: float = 1e-5,
) -> Union[float, np.ndarray]:
    """Compute the nonradiative capture coefficient.

    This function computes the nonradiative capture coefficient following the
    methodology of A. Alkauskas et al., Phys. Rev. B 90, 075202 (2014). The
    resulting capture coefficient is unscaled [See Eq. (22) of the above
    reference]. Our code assumes harmonic potential energy surfaces.

    Parameters
    ----------
    dQ : float
        displacement between harmonic oscillators in amu^{1/2} Angstrom
    dE : float
        energy offset between the two harmonic oscillators
    wi, wf : float
        frequencies of the harmonic oscillators in eV
    Wif : float
        electron-phonon coupling matrix element in eV amu^{-1/2} Angstrom^{-1}
    volume : float
        volume of the supercell in Å^3
    g : int
        degeneracy factor of the final state
    T : float, np.array(dtype=float)
        temperature or a np.array of temperatures in K
    sigma : 'pchip', 'cubic', or float
        smearing parameter in eV for replacement of the delta functions with
        gaussians. A value of 'pchip' or 'cubic' corresponds to interpolation
        instead of gaussian smearing, utilizing PCHIP or cubic spline
        interpolaton. PCHIP is preferred to cubic spline as cubic spline can
        result in negative values when small rates are found. The default is
        'pchip' and is recommended for improved accuracy.
    occ_tol : float
        criteria to determine the maximum quantum number for overlaps based on
        the Bose weights

    Returns
    -------
    float, np.array(dtype=float)
        resulting capture coefficient (unscaled) in cm^3 s^{-1}
    """
    kT = (const.k / const.e) * T     # [(J / K) * (eV / J)] * K = eV
    Z = 1. / (1 - np.exp(-wi / kT))  # Partion funtion

    Ni, Nf = (17, 50)   # default values
    tNi = np.ceil(-np.max(kT) * np.log(occ_tol) / wi).astype(int)
    if tNi > Ni:
        Ni = tNi
    tNf = np.ceil((dE + Ni*wi) / wf).astype(int)
    if tNf > Nf:
        Nf = tNf
    
    # precompute values of the overlap
    ovl = np.zeros((Ni, Nf), dtype=np.longdouble)
    for m in np.arange(Ni):
        for n in np.arange(Nf):
            ovl[m, n] = overlap_NM(dQ, wi, wf, m, n)  

    t = np.linspace(-Ni*wi, Nf*wf, 5000)
    R = 0.
    for m in np.arange(Ni-1):
        weight_m = np.exp(-m * wi / kT) / Z
        if isinstance(sigma, str):
            # interpolation to replace delta functions
            E, matels = (np.zeros(Nf), np.zeros(Nf))
            for n in np.arange(Nf):
                matel = np.sqrt(Factor3) * dQ * ovl[m, n]
                E[n] = n*wf - m*wi
                matels[n] = np.abs(np.conj(matel) * matel)
            if sigma[0].lower() == 'c':
                f = interp1d(E, matels, kind='cubic', bounds_error=False,
                             fill_value=0.)
            else:
                f = PchipInterpolator(E, matels, extrapolate=False)
            R = R + weight_m * (f(dE) * np.sum(matels)
                                / np.trapz(np.nan_to_num(f(t)), x=t))
        else:
            for n in np.arange(Nf):
                # energy conservation delta function
                delta = np.exp((dE+n*wf-m*wi)/(2.0*sigma**2)) / \
                    (sigma*np.sqrt(2.0*np.pi))    ## delta(E(wf)-E(wi)) & dE (Et-EF) term 추가 
                matel = np.sqrt(Factor3) *  ovl[m, n]
                R = R + weight_m * delta * np.abs(np.conj(matel) * matel)   

    return R

if __name__ == '__main__':
    
    ## Input parameter ##
    #dQ = 0.62
    #dE = 0.0
    #wi = 0.06384
    #wf = 0.06546
    dQ = 1.68
    dE = 0.0
    wi = 0.02821
    wf = 0.03155


    Wif = 1
    volume = 1
    g = 1
    T = np.linspace(5, 2000, 400)

    # Range for computing overlaps in overlap_NM
    QQ = np.linspace(-30., 30., 5000)

    # Sigma smearing
    sigma = 'pchip'

    # Occupation tolerance
    occ_tol = 1e-10

    # Capture_coefficients 
    capture_coefficients = np.array([get_C(dQ, dE, wi, wf, Wif, volume, g, temp, sigma, occ_tol) for temp in T])

    # Capture time
    capture_time = 1/capture_coefficients

    # Save file 
    with open('nmp.txt', 'w') as file:
        for i in range(len(T)):
            file.write(f"{T[i]:.0f}  {capture_time[i]}\n")

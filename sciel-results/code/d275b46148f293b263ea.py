"""Symbolic regression candidate using a rich separable Chebyshev product basis."""
import numpy as np


def _cheb_t0(x):
    """Chebyshev T0(x) = 1"""
    return np.ones_like(x)


def _cheb_t1(x):
    """Chebyshev T1(x) = x"""
    return x


def _cheb_t2(x):
    """Chebyshev T2(x) = 2x^2 - 1"""
    return 2.0 * x**2 - 1.0


def _cheb_t3(x):
    """Chebyshev T3(x) = 4x^3 - 3x"""
    return 4.0 * x**3 - 3.0 * x


def _cheb_t4(x):
    """Chebyshev T4(x) = 8x^4 - 8x^2 + 1"""
    return 8.0 * x**4 - 8.0 * x**2 + 1.0


def _cheb_t5(x):
    """Chebyshev T5(x) = 16x^5 - 20x^3 + 5x"""
    return 16.0 * x**5 - 20.0 * x**3 + 5.0 * x


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate using a dense separable Chebyshev product basis.
    
    We allocate 10 parameters across a richer set of product terms to capture
    more complex interactions and higher-order behavior:
    
    1.  T0(x)*T0(y)   = 1
    2.  T1(x)*T0(y)   = x
    3.  T0(x)*T1(y)   = y
    4.  T2(x)*T0(y)   = 2x^2-1
    5.  T0(x)*T2(y)   = 2y^2-1
    6.  T1(x)*T1(y)   = x*y
    7.  T2(x)*T1(y)   = (2x^2-1)*y
    8.  T1(x)*T2(y)   = x*(2y^2-1)
    9.  T3(x)*T0(y)   = 4x^3-3x
    10. T0(x)*T3(y)   = 4y^3-3y
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    p = np.asarray(params).ravel()
    # Ensure exactly 10 parameters
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    else:
        p = p[:10]
    
    # Compute Chebyshev polynomials up to degree 5 for potential future use,
    # but here we specifically use up to degree 3 for the 10 selected terms.
    T0x = _cheb_t0(x)
    T1x = _cheb_t1(x)
    T2x = _cheb_t2(x)
    T3x = _cheb_t3(x)
    
    T0y = _cheb_t0(y)
    T1y = _cheb_t1(y)
    T2y = _cheb_t2(y)
    T3y = _cheb_t3(y)
    
    # Construct the basis matrix implicitly to avoid large intermediate arrays if possible,
    # but direct computation is efficient for N=200.
    
    term0 = T0x * T0y
    term1 = T1x * T0y
    term2 = T0x * T1y
    term3 = T2x * T0y
    term4 = T0x * T2y
    term5 = T1x * T1y
    term6 = T2x * T1y
    term7 = T1x * T2y
    term8 = T3x * T0y
    term9 = T0x * T3y
    
    # Linear combination
    pred = (p[0] * term0 + p[1] * term1 + p[2] * term2 + p[3] * term3 +
            p[4] * term4 + p[5] * term5 + p[6] * term6 + p[7] * term7 +
            p[8] * term8 + p[9] * term9)
    
    return pred

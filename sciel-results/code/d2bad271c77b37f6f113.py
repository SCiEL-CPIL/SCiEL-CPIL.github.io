import numpy as np
from scipy.optimize import minimize
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Models acceleration using Chebyshev polynomial basis functions.
    This provides a better-conditioned optimization landscape compared to
    raw monomials, which can suffer from ill-conditioning in BFGS.
    
    The basis functions are products of Chebyshev polynomials T_n(x) and T_m(v)
    where x and v are normalized to [-1, 1] based on training ranges.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to [-1, 1] based on training data ranges
    # x range: [-0.5728, 0.5275] -> mid: -0.02265, half_range: 0.55015
    # v range: [-0.1754, 0.2821] -> mid: 0.05335, half_range: 0.22875
    
    x_mid = -0.02265
    x_half = 0.55015
    v_mid = 0.05335
    v_half = 0.22875
    
    xn = (x - x_mid) / x_half
    vn = (v - v_mid) / v_half
    
    # Compute Chebyshev polynomials T_0, T_1, T_2, T_3
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    
    T0x = np.ones_like(xn)
    T1x = xn
    T2x = 2 * xn * xn - 1
    T3x = 4 * xn * xn * xn - 3 * xn
    
    T0v = np.ones_like(vn)
    T1v = vn
    T2v = 2 * vn * vn - 1
    T3v = 4 * vn * vn * vn - 3 * vn
    
    # Construct the model with 10 terms
    # Terms are selected to cover low-order interactions with better conditioning
    return (p0 * T0x * T0v +
            p1 * T1x * T0v +
            p2 * T2x * T0v +
            p3 * T3x * T0v +
            p4 * T0x * T1v +
            p5 * T0x * T2v +
            p6 * T0x * T3v +
            p7 * T1x * T1v +
            p8 * T2x * T1v +
            p9 * T1x * T2v)

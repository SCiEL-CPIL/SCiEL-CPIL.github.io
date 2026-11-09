"""Rewritten program for symbolic regression of a nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with driving force, from position and velocity.
    Uses Chebyshev polynomial basis for well-conditioned feature matrix,
    enabling stable BFGS optimization.
    """
    p = params
    
    # Scale inputs to [-1, 1] for Chebyshev polynomials
    # x range: ~[-0.57, 0.53], center ~ -0.02, half-range ~ 0.55
    # v range: ~[-0.18, 0.28], center ~ 0.05, half-range ~ 0.23
    
    xc = -0.02  # center of x
    xh = 0.55   # half-range of x
    vc = 0.05   # center of v
    vh = 0.23   # half-range of v
    
    # Normalize to [-1, 1]
    xn = (x - xc) / xh
    vn = (v - vc) / vh
    
    # Clip to [-1, 1] for numerical stability
    xn = np.clip(xn, -1.0, 1.0)
    vn = np.clip(vn, -1.0, 1.0)
    
    # Chebyshev polynomials of the first kind
    # T0 = 1
    # T1 = x
    # T2 = 2x^2 - 1
    # T3 = 4x^3 - 3x
    # T4 = 8x^4 - 8x^2 + 1
    
    # For x (position)
    Tx0 = np.ones_like(xn)
    Tx1 = xn
    Tx2 = 2.0 * xn**2 - 1.0
    Tx3 = 4.0 * xn**3 - 3.0 * xn
    Tx4 = 8.0 * xn**4 - 8.0 * xn**2 + 1.0
    
    # For v (velocity)
    Tv0 = np.ones_like(vn)
    Tv1 = vn
    Tv2 = 2.0 * vn**2 - 1.0
    Tv3 = 4.0 * vn**3 - 3.0 * vn
    Tv4 = 8.0 * vn**4 - 8.0 * vn**2 + 1.0
    
    # Cross terms using products of Chebyshev polynomials
    Cxv1 = xn * vn  # T1(x) * T1(v)
    Cxv2 = Tx2 * Tv1  # T2(x) * T1(v)
    
    # Combine with 10 parameters
    # p[0]: T1(x) - linear restoring
    # p[1]: T3(x) - cubic restoring (Duffing-like)
    # p[2]: T1(v) - linear damping
    # p[3]: T3(v) - cubic damping
    # p[4]: T2(x) - quadratic in x (asymmetry)
    # p[5]: T2(v) - quadratic in v
    # p[6]: T1(x)*T1(v) - linear coupling
    # p[7]: T2(x)*T1(v) - nonlinear coupling
    # p[8]: T4(x) - higher-order restoring
    # p[9]: bias
    
    y = (p[0] * Tx1 +
         p[1] * Tx3 +
         p[2] * Tv1 +
         p[3] * Tv3 +
         p[4] * Tx2 +
         p[5] * Tv2 +
         p[6] * Cxv1 +
         p[7] * Cxv2 +
         p[8] * Tx4 +
         p[9] * Tx0)
    
    return y
# EVOLVE-BLOCK-END

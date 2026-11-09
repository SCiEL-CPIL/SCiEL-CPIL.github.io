# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model using Chebyshev basis in normalized
    state variables (x, v) plus a linear time drive term.
    
    The Chebyshev polynomial basis provides superior conditioning and global
    approximation properties on bounded domains compared to monomial polynomials.
    We use degree-2 Chebyshev polynomials in normalized x and v (5 terms each),
    plus one linear time term, for a total of 10 parameters.
    
    Inputs:
        t: time (range [30, 50])
        x: position (range [-0.2056, 0.1749])
        v: velocity (range [-0.3391, 0.3624])
        params: 10 fitted constants
    
    Returns:
        dv/dt (acceleration) for each row
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Normalize x and v to [-1, 1] for Chebyshev evaluation
    # x range: [-0.2056, 0.1749], center = -0.01535, half_range = 0.19025
    x_center = -0.01535
    x_half = 0.19025
    xn = (x - x_center) / x_half
    
    # v range: [-0.3391, 0.3624], center = 0.01165, half_range = 0.35075
    v_center = 0.01165
    v_half = 0.35075
    vn = (v - v_center) / v_half
    
    # Clip to [-1, 1] to avoid numerical issues outside the training range
    xn = np.clip(xn, -1.0, 1.0)
    vn = np.clip(vn, -1.0, 1.0)
    
    # Chebyshev polynomials of degree 0, 1, 2
    # T0(u) = 1
    # T1(u) = u
    # T2(u) = 2u^2 - 1
    
    # For x (5 terms): T0, T1, T2 in x, and cross terms with v
    Tx0 = np.ones_like(xn)
    Tx1 = xn
    Tx2 = 2.0 * xn * xn - 1.0
    
    Tv0 = np.ones_like(vn)
    Tv1 = vn
    Tv2 = 2.0 * vn * vn - 1.0
    
    # Time term: linear in centered t
    tc = (t - 40.0) / 10.0
    
    # 10 basis functions:
    # 0: Tx0 (bias)
    # 1: Tx1 (linear x)
    # 2: Tx2 (quadratic x via Chebyshev)
    # 3: Tv0 * Tx1 = x (redundant, use cross term instead)
    # Let's use a better set:
    # 0: 1 (bias)
    # 1: Tx1 (x normalized)
    # 2: Tx2 (x^2 Chebyshev)
    # 3: Tv1 (v normalized)
    # 4: Tv2 (v^2 Chebyshev)
    # 5: Tx1 * Tv1 (x*v cross)
    # 6: Tx2 * Tv1 (x^2*v)
    # 7: Tx1 * Tv2 (x*v^2)
    # 8: Tx2 * Tv2 (x^2*v^2)
    # 9: tc (time drive)
    
    b0 = Tx0
    b1 = Tx1
    b2 = Tx2
    b3 = Tv1
    b4 = Tv2
    b5 = Tx1 * Tv1
    b6 = Tx2 * Tv1
    b7 = Tx1 * Tv2
    b8 = Tx2 * Tv2
    b9 = tc
    
    result = (p[0] * b0 +
              p[1] * b1 +
              p[2] * b2 +
              p[3] * b3 +
              p[4] * b4 +
              p[5] * b5 +
              p[6] * b6 +
              p[7] * b7 +
              p[8] * b8 +
              p[9] * b9)
    
    # Safety: replace any non-finite values with 0
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END

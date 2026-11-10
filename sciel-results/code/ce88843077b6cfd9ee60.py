"""Symbolic regression with 1 input using a soft-thresholded Chebyshev basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: y = sum_{i=0}^{9} params[i] * basis_i(x)
    
    Basis functions (10 total) - Soft-thresholded Chebyshev polynomials:
    The Chebyshev polynomials T_k(t) are scaled to t = x/3 in [-1,1].
    Each term is multiplied by a soft-clipping factor that provides
    piecewise-linear-like flexibility for kinks and saturation.
    
    Basis:
    0: T_0(t) * clip_0(t)
    1: T_1(t) * clip_1(t)
    2: T_2(t) * clip_2(t)
    3: T_3(t) * clip_3(t)
    4: T_4(t) * clip_4(t)
    5: T_5(t) * clip_5(t)
    6: sin(pi*t) * clip_6(t)
    7: cos(pi*t) * clip_7(t)
    8: tanh(2*t) * clip_8(t)
    9: exp(-t^2) * clip_9(t)
    
    where clip_i(t) = 1 + tanh(5.0 * (|t| - c_i)) with fixed c_i values.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Scale x from [-3, 3] to t in [-1, 1]
    t = x / 3.0
    t = np.clip(t, -1.0, 1.0)
    abs_t = np.abs(t)
    
    # Chebyshev polynomials T_k(t)
    T0 = np.ones_like(t)
    T1 = t.copy()
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    T5 = 16.0 * t**5 - 20.0 * t**3 + 5.0 * t
    
    # Trigonometric and other smooth terms
    sin_pi_t = np.sin(np.pi * t)
    cos_pi_t = np.cos(np.pi * t)
    tanh_2t = np.tanh(2.0 * t)
    exp_neg_t2 = np.exp(-t**2)
    
    # Soft-clipping factors with fixed centers c_i
    # These create piecewise-linear-like behavior around |t| = c_i
    # c_i chosen to spread the kink points across [-1, 1]
    c = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.3, 0.5, 0.7, 1.0])
    
    # Compute soft-clipping factors: 1 + tanh(5.0 * (|t| - c_i))
    # For |t| < c_i: factor is close to 1 + tanh(negative) ≈ 1 - 1 = 0 (suppressed)
    # For |t| > c_i: factor is close to 1 + tanh(positive) ≈ 1 + 1 = 2 (amplified)
    # This creates a smooth transition that can model kinks/saturation
    p = 5.0  # sharpness parameter
    clip = 1.0 + np.tanh(p * (abs_t[:, np.newaxis] - c[np.newaxis, :]))
    
    # Base functions
    base = np.column_stack([
        T0, T1, T2, T3, T4, T5,
        sin_pi_t, cos_pi_t,
        tanh_2t, exp_neg_t2
    ])  # shape (n, 10)
    
    # Apply soft-thresholding: each column multiplied by its clip factor
    basis = base * clip  # shape (n, 10)
    
    # Linear combination
    y = basis @ params
    
    # Ensure finite output
    y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return y
# EVOLVE-BLOCK-END

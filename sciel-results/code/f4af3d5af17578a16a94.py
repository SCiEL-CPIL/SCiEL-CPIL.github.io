"""Nonlinear oscillator acceleration model with Fourier time features and polynomial phase-plane basis."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model for a driven nonlinear oscillator's acceleration.
    
    Combines:
    - Fourier features in time (sin(t), cos(t)) for periodic driving
    - Polynomial basis in (x, v) phase plane for nonlinear restoring/damping
    
    Parameters (10 total):
        p[0]: amplitude of sin(t)
        p[1]: amplitude of cos(t)
        p[2]: bias (constant term)
        p[3]: coefficient of x
        p[4]: coefficient of x^2
        p[5]: coefficient of x^3
        p[6]: coefficient of v
        p[7]: coefficient of v^2
        p[8]: coefficient of x*v
        p[9]: coefficient of x^2*v + x*v^2 (cubic cross terms)
    
    Inputs:
        t: time (range [30, 50])
        x: position (range [-0.2056, 0.1749])
        v: velocity (range [-0.3391, 0.3624])
        params: 10 fitted constants
    
    Returns:
        Acceleration prediction for each row
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Fourier time features
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Polynomial phase-plane basis
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    xv = x * v
    x2v = x2 * v
    xv2 = x * v2
    
    # Combine all terms
    result = (p[0] * sin_t +
              p[1] * cos_t +
              p[2] * 1.0 +
              p[3] * x +
              p[4] * x2 +
              p[5] * x3 +
              p[6] * v +
              p[7] * v2 +
              p[8] * xv +
              p[9] * (x2v + xv2))
    
    # Safety: replace non-finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result

# EVOLVE-BLOCK-END

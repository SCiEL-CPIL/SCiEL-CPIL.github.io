"""Nonlinear oscillator acceleration model with fitted-frequency Fourier drive and Chebyshev phase-plane basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Normalization constants for x and v (center and scale to [-1, 1])
_X_CENTER = -0.0154   # (−0.2056 + 0.1749)/2
_X_SCALE = 0.1903     # (0.2056 + 0.1749)/2
_V_CENTER = 0.01165   # (−0.3391 + 0.3624)/2
_V_SCALE = 0.35075    # (0.3391 + 0.3624)/2

def _chebyshev_basis(z: np.ndarray, n: int = 4) -> np.ndarray:
    """Return Chebyshev polynomials T_0(z) through T_{n-1}(z)."""
    T = np.empty((n,) + z.shape, dtype=np.float64)
    T[0] = 1.0
    if n > 1:
        T[1] = z
    for k in range(2, n):
        T[k] = 2.0 * z * T[k-1] - T[k-2]
    return T

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model for a driven nonlinear oscillator's acceleration.
    
    Combines:
    - Fitted-frequency Fourier features: sin(omega*t + phi), cos(omega*t + phi)
    - Chebyshev polynomial basis in normalized (x, v) phase plane
    
    Parameters (10 total):
        p[0]: drive frequency omega
        p[1]: drive phase phi
        p[2]: amplitude of sin(omega*t + phi)
        p[3]: amplitude of cos(omega*t + phi)
        p[4]: bias (constant term)
        p[5]: Chebyshev coeff for x (T_1)
        p[6]: Chebyshev coeff for x^2 (T_2)
        p[7]: Chebyshev coeff for v (T_1)
        p[8]: Chebyshev coeff for v^2 (T_2)
        p[9]: Chebyshev coeff for x*v cross term
    
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
    
    # Fitted-frequency Fourier time features
    omega = p[0]
    phi = p[1]
    phase = omega * t + phi
    sin_drive = np.sin(phase)
    cos_drive = np.cos(phase)
    
    # Normalize x and v to approximately [-1, 1]
    xn = (x - _X_CENTER) / _X_SCALE
    vn = (v - _V_CENTER) / _V_SCALE
    
    # Chebyshev basis in normalized coordinates (up to degree 2)
    # T_0 = 1, T_1 = z, T_2 = 2z^2 - 1
    T1_x = xn
    T2_x = 2.0 * xn * xn - 1.0
    T1_v = vn
    T2_v = 2.0 * vn * vn - 1.0
    
    # Cross term: T_1(x) * T_1(v)
    cross = xn * vn
    
    # Combine all terms
    result = (p[2] * sin_drive +
              p[3] * cos_drive +
              p[4] * 1.0 +
              p[5] * T1_x +
              p[6] * T2_x +
              p[7] * T1_v +
              p[8] * T2_v +
              p[9] * cross)
    
    # Safety: replace non-finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result

# EVOLVE-BLOCK-END

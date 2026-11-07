"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev polynomials (T0-T4) + learnable-frequency sine + learnable cosine + learnable exponential.
    
    params[0:5] : coefficients for Chebyshev T0(z) through T4(z)
    params[5]   : frequency w for sin(w*z)
    params[6]   : amplitude A for A*sin(w*z)
    params[7]   : frequency v for cos(v*z)
    params[8]   : amplitude C for C*cos(v*z)
    params[9]   : rate r for exp(r*z) (amplitude fixed to 1.0, or could be absorbed)
    
    Wait, we have 10 params. Let's allocate:
    params[0:5] : Chebyshev coeffs
    params[5]   : sin freq
    params[6]   : sin amp
    params[7]   : cos freq
    params[8]   : cos amp
    params[9]   : exp rate (with unit amplitude)
    """
    z = x / 3.0  # Map [-3, 3] to [-1, 1]
    
    # --- Chebyshev polynomials T0..T4 on z in [-1, 1] ---
    T0 = np.ones_like(z)
    T1 = z
    T2 = 2.0 * z * z - 1.0
    T3 = 4.0 * z * z * z - 3.0 * z
    T4 = 8.0 * z * z * z * z - 8.0 * z * z + 1.0
    
    # --- Learnable-frequency sinusoidal term ---
    w = params[5]
    A = params[6]
    # Clip frequency to avoid numerical issues and ensure smoothness
    w_safe = np.clip(w, -20.0, 20.0)
    sin_term = A * np.sin(w_safe * z)
    
    # --- Learnable-frequency cosine term ---
    v = params[7]
    C = params[8]
    v_safe = np.clip(v, -20.0, 20.0)
    cos_term = C * np.cos(v_safe * z)
    
    # --- Learnable exponential term ---
    r = params[9]
    # Clip r to avoid overflow: exp(r*z) with z in [-1,1], r up to ~20 is safe
    r_safe = np.clip(r, -10.0, 10.0)
    exp_term = 1.0 * np.exp(r_safe * z)
    
    # --- Assemble prediction ---
    pred = (params[0] * T0 +
            params[1] * T1 +
            params[2] * T2 +
            params[3] * T3 +
            params[4] * T4 +
            sin_term +
            cos_term +
            exp_term)
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END

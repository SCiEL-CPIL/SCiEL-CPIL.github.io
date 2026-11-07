"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE_SECONDS = 140.0
_START = time.time()


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator.
    
    Combines the physical structure of a Duffing-type oscillator with 
    normalized inputs to ensure well-conditioned optimization for BFGS.
    
    Basis functions:
    1. Constant
    2. sin(t)
    3. cos(t)
    4. x (normalized)
    5. x^2 (normalized)
    6. x^3 (normalized)
    7. v (normalized)
    8. v^2 (normalized)
    9. v^3 (normalized)
    10. x*v (normalized)
    
    Normalization:
    t_centered = (t - 40) / 10  -> range [-1, 1] approx
    x_norm = x / 0.35           -> range approx [-1, 1]
    v_norm = v / 0.70           -> range approx [-1, 1]
    """
    if time.time() > _DEADLINE_SECONDS:
        return np.zeros_like(np.asarray(t, dtype=float), dtype=float)

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    
    # Normalize inputs to have comparable scales for BFGS stability
    # Training ranges: t [30, 50], x [-0.34, 0.36], v [-0.71, 0.66]
    t_n = (t - 40.0) / 10.0
    x_n = x / 0.35
    v_n = v / 0.70

    # Precompute powers and trig terms
    x2 = x_n * x_n
    x3 = x2 * x_n
    v2 = v_n * v_n
    v3 = v2 * v_n
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    x_v = x_n * v_n

    # Linear combination of 10 basis functions
    # params: [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]
    return (
        params[0] * 1.0
        + params[1] * sin_t
        + params[2] * cos_t
        + params[3] * x_n
        + params[4] * x2
        + params[5] * x3
        + params[6] * v_n
        + params[7] * v2
        + params[8] * v3
        + params[9] * x_v
    )
# EVOLVE-BLOCK-END

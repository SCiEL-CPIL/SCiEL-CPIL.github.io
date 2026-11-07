"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev polynomials + trigonometric terms.
    Uses 10 parameters linearly combined with a rich basis to capture
    both polynomial and oscillatory components of the target function.
    """
    t_start = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    t = x / 2.0  # Map [-2, 2] to [-1, 1]
    
    # Chebyshev polynomials T_0 ... T_4
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    
    # Trigonometric terms at various frequencies
    sin1 = np.sin(np.pi * t)
    cos1 = np.cos(np.pi * t)
    sin2 = np.sin(2.0 * np.pi * t)
    cos2 = np.cos(2.0 * np.pi * t)
    sin3 = np.sin(3.0 * np.pi * t)
    cos3 = np.cos(3.0 * np.pi * t)
    
    # Linear combination
    return (params[0] * T0 +
            params[1] * T1 +
            params[2] * T2 +
            params[3] * T3 +
            params[4] * T4 +
            params[5] * sin1 +
            params[6] * cos1 +
            params[7] * sin2 +
            params[8] * cos2 +
            params[9] * sin3)
# EVOLVE-BLOCK-END

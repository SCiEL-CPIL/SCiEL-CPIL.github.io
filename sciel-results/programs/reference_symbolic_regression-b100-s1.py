"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(x)

    # Use a combined basis: Chebyshev polynomials (for smooth, non-oscillatory features)
    # and Fourier terms (for oscillatory features).
    # With 10 params, we'll use:
    # - 5 Chebyshev polynomials T_0..T_4 (scaled to [-1,1])
    # - 5 Fourier terms: sin(pi*x), cos(pi*x), sin(2*pi*x), cos(2*pi*x), sin(3*pi*x)
    #   Wait, that's 5 Fourier terms. Total = 5 + 5 = 10.
    # Actually, let's be more careful. Let's use:
    # T_0, T_1, T_2, T_3, T_4 (5 Chebyshev)
    # sin(pi*x), cos(pi*x), sin(2*pi*x), cos(2*pi*x), sin(3*pi*x) (5 Fourier)
    # Total = 10 terms.

    n = len(x)
    if n == 0:
        return np.array([])

    # Map x to [-1, 1] for Chebyshev
    t = x / 2.0

    # Compute Chebyshev polynomials T_0 through T_4
    T0 = np.ones(n)
    T1 = t.copy()
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2

    # Compute Fourier terms
    # Use frequencies that are "natural" for the domain [-2, 2]
    # pi*x gives one full period over [-2,2]
    # 2*pi*x gives two full periods
    # 3*pi*x gives three full periods
    F1 = np.sin(np.pi * x)
    F2 = np.cos(np.pi * x)
    F3 = np.sin(2.0 * np.pi * x)
    F4 = np.cos(2.0 * np.pi * x)
    F5 = np.sin(3.0 * np.pi * x)

    # Combine: params[0..4] for Chebyshev, params[5..9] for Fourier
    result = (params[0] * T0 +
              params[1] * T1 +
              params[2] * T2 +
              params[3] * T3 +
              params[4] * T4 +
              params[5] * F1 +
              params[6] * F2 +
              params[7] * F3 +
              params[8] * F4 +
              params[9] * F5)

    return result
# EVOLVE-BLOCK-END

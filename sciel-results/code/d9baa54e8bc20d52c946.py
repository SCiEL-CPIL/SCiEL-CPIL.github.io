"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a hybrid Chebyshev + Fourier basis.
    
    The 10 parameters are split:
    - params[0:4] : coefficients for Chebyshev polynomials T0-T3 (global trend)
    - params[4:10]: coefficients for sin/cos terms at frequencies 1, 2, 3 (oscillations)
    
    This hybrid captures both smooth global structure and periodic/local features.
    """
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)

    # Normalize to [-1, 1] for Chebyshev stability
    t = x / 3.0

    # --- Chebyshev polynomials T0 to T3 ---
    b0 = np.ones(n)          # T_0
    b1 = t.copy()            # T_1
    b2 = 2.0 * t**2 - 1.0    # T_2
    b3 = 4.0 * t**3 - 3.0 * t  # T_3

    # --- Fourier terms: sin/cos at x, 2x, 3x ---
    b4 = np.sin(x)
    b5 = np.cos(x)
    b6 = np.sin(2.0 * x)
    b7 = np.cos(2.0 * x)
    b8 = np.sin(3.0 * x)
    b9 = np.cos(3.0 * x)

    # Stack basis and compute linear combination
    basis = np.vstack([
        b0, b1, b2, b3,
        b4, b5, b6, b7,
        b8, b9
    ])

    result = basis.T @ params

    # Safety: ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)

    if time.time() > deadline:
        pass  # already computed, just return

    return result
# EVOLVE-BLOCK-END

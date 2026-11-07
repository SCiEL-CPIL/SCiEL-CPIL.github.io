"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 145.0  # safety margin before 150s kill


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Full quadratic polynomial in centered x, y, z with 10 parameters.
    Centering around 1.25 (midpoint of [0.5, 2]) improves Hessian conditioning
    for BFGS convergence.
    params[0]  : constant
    params[1]  : xc = x - 1.25
    params[2]  : yc = y - 1.25
    params[3]  : zc = z - 1.25
    params[4]  : xc^2
    params[5]  : yc^2
    params[6]  : zc^2
    params[7]  : xc*yc
    params[8]  : xc*zc
    params[9]  : yc*zc
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # Center inputs around midpoint of [0.5, 2]
    xc = x - 1.25
    yc = y - 1.25
    zc = z - 1.25

    n = x.size
    # Build feature matrix: [1, xc, yc, zc, xc^2, yc^2, zc^2, xc*yc, xc*zc, yc*zc]
    features = np.empty((n, 10), dtype=np.float64)
    features[:, 0] = 1.0
    features[:, 1] = xc
    features[:, 2] = yc
    features[:, 3] = zc
    features[:, 4] = xc * xc
    features[:, 5] = yc * yc
    features[:, 6] = zc * zc
    features[:, 7] = xc * yc
    features[:, 8] = xc * zc
    features[:, 9] = yc * zc

    # Single BLAS dgemv call
    return features @ p
# EVOLVE-BLOCK-END

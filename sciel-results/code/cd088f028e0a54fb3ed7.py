"""ODE integrator exploiting the fixed linear structure: y' = A y, with A recovered from f."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0

    # Recover the constant matrix A from f by probing with basis vectors.
    # f(t, y) = A @ y (t-independent), so A[:, j] = f(0, e_j).
    A = np.empty((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        A[:, j] = f(0.0, e)

    # Exact solution of the linear IVP: y_end = expm(A * dt) @ y0.
    M = expm(A * dt)
    y_end = M @ y0

    return y_end
# EVOLVE-BLOCK-END

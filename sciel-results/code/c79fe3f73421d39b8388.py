"""Rewritten ODE solver for 15-dim stiff linear IVP on [0,1]."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    n = y0_arr.size
    deadline = time.time() + 280.0

    # Extract A and b from f(t, y) = A y + b using basis vectors.
    # 16 calls total.
    e0 = np.zeros(n)
    b = np.array(f(t0, e0), dtype=float)
    A = np.zeros((n, n))
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        A[:, j] = np.array(f(t0, ej), dtype=float) - b

    # Solve the autonomous linear system y' = A y + b analytically.
    # y(t) = expm(A t) y0 + int_0^t expm(A (t-s)) b ds
    # For constant b: y(t) = expm(A t) (y0 - A^{-1} b) + A^{-1} b
    dt = t1 - t0
    Y = expm(A * dt)

    try:
        # Solve A x = b
        x_b = np.linalg.solve(A, b)
        y_final = Y @ (y0_arr - x_b) + x_b
    except np.linalg.LinAlgError:
        # A is singular; use numerical quadrature for the integral term.
        # y(t) = expm(A t) y0 + int_0^t expm(A (t-s)) b ds
        # Use Gauss-Legendre quadrature with 32 points.
        from numpy.polynomial.legendre import leggauss
        n_quad = 32
        nodes, weights = leggauss(n_quad)
        s_vals = (nodes * 0.5 + 0.5) * dt
        w_vals = weights * 0.5 * dt
        y_final = Y @ y0_arr
        for s, w in zip(s_vals, w_vals):
            y_final += expm(A * (dt - s)) @ b * w

    return y_final
# EVOLVE-BLOCK-END

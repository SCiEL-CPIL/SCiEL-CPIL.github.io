"""ODE solver: 15-dimensional stiff linear system on [0,1]."""
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

    dt = t1 - t0

    # Primary approach: use smaller (n x n) expm + linear solve.
    # y(t) = expm(A*t) * (y0 - A^{-1} b) + A^{-1} b
    # This is exact for the autonomous linear system.
    Y = expm(A * dt)

    try:
        # Check condition number to decide if solve is safe
        cond_A = np.linalg.cond(A)
        if cond_A < 1e12:
            x_b = np.linalg.solve(A, b)
            y_final = Y @ (y0_arr - x_b) + x_b
        else:
            # A is ill-conditioned; use augmented matrix exponential
            M = np.zeros((n + 1, n + 1))
            M[:n, :n] = A
            M[:n, n] = -b
            M[n, n] = 1.0
            E = expm(M * dt)
            y_final = E[:n, :n] @ y0_arr + E[:n, n]
    except np.linalg.LinAlgError:
        # A is singular; use augmented matrix exponential
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = A
        M[:n, n] = -b
        M[n, n] = 1.0
        E = expm(M * dt)
        y_final = E[:n, :n] @ y0_arr + E[:n, n]

    # High-precision polish: verify with 2 substeps of exact flow map.
    # For a linear autonomous system, this should agree to machine precision.
    # If there's any discrepancy (due to numerical issues in expm for large
    # ||A||*dt), the substep approach may be more accurate.
    if time.time() < deadline - 10.0:
        h = dt / 2.0
        M_half = expm(A * h)
        # For constant b: y(t+h) = expm(A*h)*y + (expm(A*h) - I) * A^{-1} * b
        # But if A is singular, use the integral form via augmented matrix
        try:
            if np.linalg.cond(A) < 1e12:
                x_b = np.linalg.solve(A, b)
                c_half = M_half @ x_b - x_b  # = (M_half - I) @ x_b
            else:
                M_aug_half = np.zeros((n + 1, n + 1))
                M_aug_half[:n, :n] = A
                M_aug_half[:n, n] = -b
                M_aug_half[n, n] = 1.0
                E_half = expm(M_aug_half * h)
                M_half = E_half[:n, :n]
                c_half = E_half[:n, n]

            y_check = y0_arr.copy()
            y_check = M_half @ y_check + c_half
            y_check = M_half @ y_check + c_half

            # Compare; if substep result is finite and differs significantly,
            # prefer the substep result (more numerically stable for stiff systems)
            if np.all(np.isfinite(y_check)):
                diff = np.linalg.norm(y_check - y_final)
                scale = max(np.linalg.norm(y_final), np.linalg.norm(y_check), 1e-30)
                if diff > 1e-14 * scale:
                    y_final = y_check
        except Exception:
            pass  # Keep original result

    return y_final
# EVOLVE-BLOCK-END

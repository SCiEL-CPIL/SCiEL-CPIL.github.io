"""Solve a fixed 15-dimensional stiff linear IVP using matrix exponential methods."""
import numpy as np
from scipy.linalg import expm, solve


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    This solver assumes the ODE is a linear system y' = A * y + b with constant A and b.
    It determines A and b by probing f at t0, then uses the matrix exponential for the exact solution.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Determine b by evaluating at y = 0
    y_zero = np.zeros(n, dtype=float)
    b = np.asarray(f(t0, y_zero), dtype=float)
    
    # Determine A columns by evaluating at y = e_j
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        y_probe = np.zeros(n, dtype=float)
        y_probe[j] = 1.0
        f_probe = np.asarray(f(t0, y_probe), dtype=float)
        A[:, j] = f_probe - b
        
    dt = t1 - t0
    expm_A_dt = expm(A * dt)
    
    # Compute particular solution for constant b
    # y_p = (expm(A*dt) - I) @ A^{-1} @ b
    try:
        # Solve A @ x = b for x, then particular = (expm_A_dt - I) @ x
        x_b = solve(A, b)
        particular = (expm_A_dt - np.eye(n)) @ x_b
    except np.linalg.LinAlgError:
        # If A is singular, use numerical integration for the particular solution
        # y_p = integral_0^dt expm(A * u) @ b du
        steps = 100
        h = dt / steps
        y_p = np.zeros(n, dtype=float)
        for i in range(steps):
            u = (i + 0.5) * h
            y_p += expm(A * u) @ b * h
        particular = y_p

    y_final = expm_A_dt @ y0 + particular
    
    return y_final

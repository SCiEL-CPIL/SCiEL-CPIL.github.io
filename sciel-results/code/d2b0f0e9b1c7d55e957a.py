"""ODE solver: 15-dimensional stiff linear system using matrix exponential."""
import numpy as np
from scipy.linalg import expm
import time


def integrate(f, t_span, y0):
    t_start = time.time()
    deadline = t_start + 300.0  # Safety margin before 330s kill

    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0

    # Probe for linear time-invariant structure: y' = A*y + b
    # b = f(t0, 0)
    y_zero = np.zeros(n)
    b = np.array(f(t0, y_zero), dtype=float)

    # A[:, j] = f(t0, e_j)
    A = np.zeros((n, n))
    for j in range(n):
        if time.time() > deadline:
            break
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = np.array(f(t0, e_j), dtype=float)

    # Verify linearity with a single test point (reduces calls from 2 to 1)
    y_test = np.ones(n) * 2.0
    expected = A @ y_test + b
    actual = np.array(f(t0, y_test), dtype=float)
    if not np.allclose(expected, actual, rtol=1e-10, atol=1e-12):
        # Fallback: use scipy's solve_ivp with BDF method for stiff problems
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='BDF', rtol=1e-12, atol=1e-14, max_step=dt/100)
        return sol.y[:, -1]

    # Check time-invariance at a different time (1 more call)
    t_mid = (t0 + t1) / 2
    expected_mid = A @ y_test + b
    actual_mid = np.array(f(t_mid, y_test), dtype=float)
    if not np.allclose(expected_mid, actual_mid, rtol=1e-10, atol=1e-12):
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='BDF', rtol=1e-12, atol=1e-14, max_step=dt/100)
        return sol.y[:, -1]

    # Linear time-invariant system: y' = A*y + b
    # Use augmented matrix approach for exact solution
    A_aug = np.zeros((n + 1, n + 1))
    A_aug[:n, :n] = A
    A_aug[:n, n] = b
    # Last row is all zeros

    y_aug0 = np.zeros(n + 1)
    y_aug0[:n] = y0
    y_aug0[n] = 1.0

    M = expm(A_aug * dt)
    y_aug = M @ y_aug0
    y_result = y_aug[:n].copy()

    return y_result

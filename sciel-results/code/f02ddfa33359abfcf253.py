"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0

    # Probe linearity and time-independence with a few calls
    # First, check if f depends on t: compare f(t0, y0) vs f(t1, y0)
    fy_t0 = np.asarray(f(t0, y0), dtype=float)
    fy_t1 = np.asarray(f(t1, y0), dtype=float)

    # Check if f is linear: f(t, a*y1 + b*y2) == a*f(t,y1) + b*f(t,y2)
    # Use two test vectors
    v1 = y0.copy()
    if np.max(np.abs(v1)) < 1e-14:
        v1 = np.ones(n)
    v2 = np.zeros(n)
    v2[0] = 1.0
    if np.allclose(v1, v2):
        v2 = np.zeros(n)
        v2[-1] = 1.0

    a, b = 0.7, 0.3
    y_test = a * v1 + b * v2
    fy_test = np.asarray(f(t0, y_test), dtype=float)
    fy_v1 = np.asarray(f(t0, v1), dtype=float)
    fy_v2 = np.asarray(f(t0, v2), dtype=float)
    linear_check = np.allclose(fy_test, a * fy_v1 + b * fy_v2, atol=1e-10, rtol=1e-8)

    time_indep_check = np.allclose(fy_t0, fy_t1, atol=1e-10, rtol=1e-8)

    if linear_check and time_indep_check:
        # Extract matrix A: A[:, j] = f(t0, e_j)
        A = np.zeros((n, n), dtype=float)
        for j in range(n):
            ej = np.zeros(n)
            ej[j] = 1.0
            A[:, j] = np.asarray(f(t0, ej), dtype=float)
        # Compute y(t1) = expm(A * dt) @ y0
        M = A * dt
        # Use expm for accuracy
        E = expm(M)
        y_end = E @ y0
        return y_end
    else:
        # Fallback: use a high-order implicit method
        # Use scipy's solve_ivp with BDF or Radau for stiff problems
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, [t0, t1], y0, method='Radau', rtol=1e-12, atol=1e-14, max_step=dt / 500)
        if sol.success:
            return sol.y[:, -1]
        # Last resort: RK4 with many steps
        steps = 5000
        h = dt / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y
# EVOLVE-BLOCK-END

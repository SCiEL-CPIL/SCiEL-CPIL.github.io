import numpy as np
from scipy.linalg import expm
import time


def _extract_A(f, t, n):
    """Extract A(t) assuming f(t,y) = A(t) y (homogeneous)."""
    A = np.empty((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A[:, j] = f(t, e)
    return A


def _is_linear_time_indep(f, t0, t1, y0, A, n, tol=1e-9):
    """Check if f(t,y) = A*y for all t in [t0,t1] and all y (linear, homogeneous, time-independent)."""
    # Homogeneity: f(t0, 0) ≈ 0
    z0 = f(t0, np.zeros(n))
    hom_err = np.linalg.norm(z0)
    if hom_err > 1e-10:
        return False

    # Linearity probes at t0 and t1
    base = y0.copy()
    if np.linalg.norm(base) < 1e-14:
        base = np.ones(n, dtype=float)
    probes = [base, 2.7 * base, -1.3 * base, np.arange(1, n + 1, dtype=float)]
    if n >= 2:
        probes.append(base + 0.5 * np.arange(1, n + 1, dtype=float))

    maxrel = 0.0
    for py in probes:
        pred = A @ py
        fy0 = f(t0, py)
        maxrel = max(maxrel, np.linalg.norm(fy0 - pred) / (1e-300 + np.linalg.norm(fy0)))
        fy1 = f(t1, py)
        maxrel = max(maxrel, np.linalg.norm(fy1 - pred) / (1e-300 + np.linalg.norm(fy1)))

    return maxrel < tol


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = y0.size
    dt = t1 - t0
    start = time.time()
    deadline = start + 290.0  # 10s safety margin

    if dt == 0:
        return y0.copy()

    # --- Attempt 1: detect a time-independent linear (homogeneous) system f(t, y) = A y.
    A = _extract_A(f, t0, n)

    if _is_linear_time_indep(f, t0, t1, y0, A, n, tol=1e-9):
        # Constant linear system: y(t1) = expm(A dt) @ y0.
        M = expm(A * dt)
        return M @ y0

    # --- Attempt 2: detect a time-dependent linear (homogeneous) system f(t, y) = A(t) y.
    # Use a piecewise-constant approximation with K subintervals.
    # Check linearity at several time points.
    K = 16  # number of subintervals
    t_mid = 0.5 * (t0 + t1)
    A_mid = _extract_A(f, t_mid, n)

    # Check if f is linear (but possibly time-dependent) at t0, t_mid, t1
    base = y0.copy()
    if np.linalg.norm(base) < 1e-14:
        base = np.ones(n, dtype=float)
    probe = base + 0.5 * np.arange(1, n + 1, dtype=float)

    maxrel = 0.0
    for t_chk in [t0, t_mid, t1]:
        A_chk = _extract_A(f, t_chk, n)
        pred = A_chk @ probe
        fy = f(t_chk, probe)
        maxrel = max(maxrel, np.linalg.norm(fy - pred) / (1e-300 + np.linalg.norm(fy)))
        # Homogeneity check
        z = f(t_chk, np.zeros(n))
        maxrel = max(maxrel, np.linalg.norm(z) / (1e-300 + np.linalg.norm(fy)))

    if maxrel < 1e-9:
        # Time-dependent linear homogeneous system: use product of exponentials
        # y(t1) ≈ exp(A(t_K) * dt/K) * ... * exp(A(t_1) * dt/K) @ y0
        # Use midpoint rule for A(t) at each subinterval
        sub_dt = dt / K
        y_cur = y0.copy()
        for k in range(K):
            t_k = t0 + (k + 0.5) * sub_dt
            A_k = _extract_A(f, t_k, n)
            M_k = expm(A_k * sub_dt)
            y_cur = M_k @ y_cur
        return y_cur

    # --- Fallback: robust implicit stiff integrator.
    if time.time() > deadline - 30.0:
        # Not enough time; return a coarse explicit step result as a best effort.
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, (t0, t1), y0, method='Radau', rtol=1e-8, atol=1e-10,
                        max_step=0.1)
        return sol.y[:, -1]

    from scipy.integrate import solve_ivp
    sol = solve_ivp(f, (t0, t1), y0, method='Radau', rtol=1e-11, atol=1e-13,
                    max_step=0.05)
    if not sol.success:
        # Try BDF as a last resort.
        sol = solve_ivp(f, (t0, t1), y0, method='BDF', rtol=1e-11, atol=1e-13,
                        max_step=0.05)
    return sol.y[:, -1]

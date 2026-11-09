"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: return best result before 330s wall-clock limit
    start_time = time.time()
    deadline = start_time + 320.0  # 10s safety margin
    
    # Use Radau (implicit Runge-Kutta, L-stable) for stiff problems
    # This is highly efficient for stiff linear systems
    # rtol and atol control accuracy; lower = more accurate but more function calls
    # The score is -log10(rel_err) - 0.5*log10(nfev), so we want low error AND low nfev
    
    # Try with tight tolerances first
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # If time is running out, try a faster but slightly less accurate solve
    if time.time() > deadline - 30.0:
        try:
            sol = solve_ivp(
                f, t_span, y0,
                method='Radau',
                rtol=1e-10,
                atol=1e-12,
                dense_output=False
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Fallback: try BDF
    if time.time() < deadline:
        try:
            sol = solve_ivp(
                f, t_span, y0,
                method='BDF',
                rtol=1e-12,
                atol=1e-14,
                dense_output=False
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Last resort: manual backward Euler with few steps
    # This should rarely be reached
    n = len(y0)
    steps = 1000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        # One Newton iteration per step for implicit Euler
        for _newton in range(3):
            fz = f(t + h, y)
            J = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = 1e-7 * max(1.0, abs(y[j]))
                J[:, j] = (f(t + h, y + e) - fz) / e[j]
            step = np.linalg.solve(np.eye(n) - h * J, -(y - y0 - h * fz))
            y_new = y + step
            if np.linalg.norm(step) <= 1e-10 * (1 + np.linalg.norm(y_new)):
                y = y_new
                break
            y = y_new
        y0 = y
        t += h
        if time.time() > deadline:
            break
    return y
# EVOLVE-BLOCK-END

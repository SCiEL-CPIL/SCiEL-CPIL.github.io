"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 280.0  # Safety margin before 330s limit

def _check_deadline():
    """Return True if we should stop integrating."""
    return time.time() > _DEADLINE


def _rk4(f, t0, t1, y0, steps):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        if _check_deadline():
            return None
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y


def _backward_euler(f, t0, t1, y0, steps):
    # Implicit Euler with a finite-difference Jacobian and a few Newton iterations (handles stiff problems).
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    n = len(y)
    for _ in range(steps):
        if _check_deadline():
            return None
        z = y.copy()
        for _newton in range(4):
            fz = f(t + h, z)
            J = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = 1e-7 * max(1.0, abs(z[j]))
                J[:, j] = (f(t + h, z + e) - fz) / e[j]
            step = np.linalg.solve(np.eye(n) - h * J, -(z - y - h * fz))
            z = z + step
            if np.linalg.norm(step) <= 1e-10 * (1 + np.linalg.norm(z)):
                break
        y, t = z, t + h
    return y


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Strategy:
    1. Try adaptive RK45 (Dormand-Prince) via scipy with tight tolerances.
    2. If that fails or is too slow, fall back to stiff solvers (BDF/Radau).
    3. Final fallback: fixed-step RK4 or implicit Euler.
    
    The score is -log10(rel_err) - 0.5*log10(f_calls), so we want both small error
    and few function evaluations.
    """
    global _DEADLINE
    _DEADLINE = time.time() + 280.0
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    from scipy.integrate import solve_ivp
    
    # Method 1: RK45 (non-stiff, efficient)
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='RK45',
            rtol=1e-10, atol=1e-12,
            max_step=(t1 - t0) / 50,
            dense_output=False
        )
        if sol.success and sol.t[-1] == t1 and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass
    
    if _check_deadline():
        # Fallback to RK4
        y = _rk4(f, t0, t1, y0, 2000)
        if y is not None:
            return y
        y = _backward_euler(f, t0, t1, y0, 2000)
        if y is not None:
            return y
        return y0
    
    # Method 2: BDF (stiff-capable)
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='BDF',
            rtol=1e-10, atol=1e-12,
            max_step=(t1 - t0) / 20,
            dense_output=False
        )
        if sol.success and sol.t[-1] == t1 and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass
    
    if _check_deadline():
        y = _rk4(f, t0, t1, y0, 2000)
        if y is not None:
            return y
        y = _backward_euler(f, t0, t1, y0, 2000)
        if y is not None:
            return y
        return y0
    
    # Method 3: Radau (stiff, high order)
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-10, atol=1e-12,
            max_step=(t1 - t0) / 20,
            dense_output=False
        )
        if sol.success and sol.t[-1] == t1 and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Final fallback: RK4 with 2000 steps
    y = _rk4(f, t0, t1, y0, 2000)
    if y is not None:
        return y
    
    # Last resort: implicit Euler
    y = _backward_euler(f, t0, t1, y0, 2000)
    if y is not None:
        return y
    
    return y0
# EVOLVE-BLOCK-END

"""ODE solver for 100-point convection-diffusion IVP using matrix exponential with linearity verification."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t_start = time.perf_counter()
    DEADLINE = 280.0  # Safety margin before 300s kill

    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    t0, t1 = t_span
    T = t1 - t0

    # Extract the system matrix A by probing with basis vectors.
    # For a linear time-invariant system: f(t, y) = A @ y
    # f(t0, e_i) = A @ e_i = column i of A
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        e_i = np.zeros(n, dtype=float)
        e_i[i] = 1.0
        A[:, i] = f(t0, e_i)

    # Check for NaNs or Infs
    if not np.all(np.isfinite(A)):
        return _fallback_adaptive(f, t0, t1, y0, DEADLINE, t_start)

    # Verify linearity: check that f(t0, y0) ≈ A @ y0
    f_y0 = f(t0, y0)
    A_y0 = A @ y0
    norm_f = np.linalg.norm(f_y0)
    norm_diff = np.linalg.norm(f_y0 - A_y0)
    rel_residual = norm_diff / (norm_f + 1e-30)

    if rel_residual < 1e-10:
        # Confirmed linear time-invariant system
        # Compute y(t1) = expm(A * T) @ y0
        try:
            M = expm(A * T)
            y_end = M @ y0
            if np.all(np.isfinite(y_end)):
                return y_end
        except Exception:
            pass

    # Fallback: high-accuracy adaptive solver
    return _fallback_adaptive(f, t0, t1, y0, DEADLINE, t_start)

def _fallback_adaptive(f, t0, t1, y0, deadline, t_start):
    """High-accuracy fallback using DOP853 with tight tolerances."""
    # Try DOP853 first (explicit, high-order)
    try:
        sol = solve_ivp(
            f, (t0, t1), y0,
            method='DOP853',
            rtol=1e-13,
            atol=1e-15,
            max_step=(t1 - t0) / 100.0,
            dense_output=False
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass

    # Check deadline
    if (time.perf_counter() - t_start) > deadline - 30.0:
        # Emergency: simple RK4
        return _fallback_rk4(f, t0, t1, y0, deadline, t_start)

    # Try Radau for stiff problems
    try:
        sol = solve_ivp(
            f, (t0, t1), y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 100.0,
            dense_output=False
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: high-accuracy RK4
    return _fallback_rk4(f, t0, t1, y0, deadline, t_start)

def _fallback_rk4(f, t0, t1, y0, deadline, t_start):
    """Simple RK4 fallback with fixed steps."""
    steps = 5000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        if (time.perf_counter() - t_start) > deadline - 5.0:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END

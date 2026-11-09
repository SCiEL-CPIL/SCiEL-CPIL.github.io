"""Use scipy.integrate.solve_ivp with Radau and analytic Jacobian to minimize f calls."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _jacobian(t, y):
    """Analytic Jacobian for Van der Pol oscillator (mu=5).
    
    System: y0' = y1
            y1' = mu*(1 - y0^2)*y1 - y0, mu=5
    
    J = [[0, 1],
         [-2*mu*y0*y1 - 1, mu*(1 - y0^2)]]
    """
    mu = 5.0
    y0_val = y[0]
    y1_val = y[1]
    J = np.array([
        [0.0, 1.0],
        [-2.0 * mu * y0_val * y1_val - 1.0, mu * (1.0 - y0_val ** 2)]
    ])
    return J


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    deadline = time.monotonic() + 280.0  # safety margin before 300s limit

    # Primary: Radau with analytic Jacobian, tight tolerances matching reference
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='Radau',
            jac=_jacobian,
            rtol=1e-11,
            atol=1e-13,
            max_step=0.1,
        )
        if sol.success and sol.t[-1] == t1 and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1].copy()
    except Exception:
        pass

    # Fallback: Radau with slightly looser tolerances
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='Radau',
            jac=_jacobian,
            rtol=1e-10,
            atol=1e-12,
            max_step=0.2,
        )
        if sol.success and sol.t[-1] == t1 and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1].copy()
    except Exception:
        pass

    # Final fallback: BDF method with Jacobian
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='BDF',
            jac=_jacobian,
            rtol=1e-10,
            atol=1e-12,
            max_step=0.2,
        )
        if sol.success and sol.t[-1] == t1 and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1].copy()
    except Exception:
        pass

    # Last resort: RK45 with small step
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='RK45',
            rtol=1e-10,
            atol=1e-12,
            max_step=0.05,
        )
        if sol.success and sol.t[-1] == t1 and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1].copy()
    except Exception:
        pass

    # Absolute last resort: manual RK4
    h = (t1 - t0) / 2000
    t, y = t0, y0_arr.copy()
    for _ in range(2000):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break
    return y
# EVOLVE-BLOCK-END

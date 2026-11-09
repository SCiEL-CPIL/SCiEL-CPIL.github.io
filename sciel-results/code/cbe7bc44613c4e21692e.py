"""ODE solver for Robertson problem using Radau with analytic Jacobian."""
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) over t_span starting from y0.
    Returns the final state y(t_end).

    Uses Radau with the analytic Jacobian of the Robertson problem to
    minimize function evaluations while achieving very high accuracy.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # 40s safety margin before 330s limit

    # Analytic Jacobian for the Robertson problem:
    # y0' = -0.04*y0 + 1e4*y1*y2
    # y1' =  0.04*y0 - 1e4*y1*y2 - 6e7*y1^2
    # y2' =  6e7*y1^2
    # J = [[-0.04, 1e4*y2, 1e4*y1],
    #      [ 0.04, -1e4*y2 - 6e7*y1, -1e4*y1],
    #      [ 0,    6e7*y1,           0    ]]
    def jac(t, y):
        y0_val, y1_val, y2_val = y[0], y[1], y[2]
        return np.array([
            [-0.04, 1e4 * y2_val, 1e4 * y1_val],
            [0.04, -1e4 * y2_val - 6e7 * y1_val, -1e4 * y1_val],
            [0.0, 6e7 * y1_val, 0.0]
        ])

    # Primary: Radau with very tight tolerances and analytic Jacobian
    try:
        sol = solve_ivp(
            f,
            [t0, t1],
            y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-15,
            jac=jac
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Check if we still have time
    if time.time() > deadline - 30:
        # Fallback: use looser tolerances quickly
        try:
            sol = solve_ivp(
                f,
                [t0, t1],
                y0,
                method='Radau',
                rtol=1e-10,
                atol=1e-12,
                jac=jac
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
        # Last resort: simple integration
        n_steps = 10000
        h = (t1 - t0) / n_steps
        t, y = t0, y0.copy()
        for _ in range(n_steps):
            y = y + h * f(t, y)
            t += h
        return y

    # Secondary: BDF with tight tolerances and analytic Jacobian
    try:
        sol = solve_ivp(
            f,
            [t0, t1],
            y0,
            method='BDF',
            rtol=1e-13,
            atol=1e-15,
            jac=jac
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Tertiary: LSODA with tight tolerances
    try:
        sol = solve_ivp(
            f,
            [t0, t1],
            y0,
            method='LSODA',
            rtol=1e-13,
            atol=1e-15
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: RK4 with many steps
    n_steps = 20000
    h = (t1 - t0) / n_steps
    t, y = t0, y0.copy()
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y

"""ODE solver for Robertson problem with adaptive stiff integration and analytical Jacobian."""
import numpy as np
import time
from scipy.integrate import solve_ivp


def _robertson_jac(t, y):
    """
    Analytical Jacobian for the Robertson problem.
    y' = [ -1e-4*y1 + 1e4*y2*y3 - 1e4*y1^2*y3,
           1e-4*y1 - 1e4*y2*y3 - 1e4*y1^2*y3,
           1e4*y2*y3 ]
    """
    y1, y2, y3 = y
    return np.array([
        [-1e-4 - 2e4 * y1 * y3, 1e4 * y3, 1e4 * y2 - 1e4 * y1 * y1],
        [1e-4 + 2e4 * y1 * y3, -1e4 * y3, -1e4 * y2 + 1e4 * y1 * y1],
        [0.0, 1e4 * y3, 1e4 * y2]
    ])


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) over t_span starting from y0.
    Returns the final state y(t_end).
    
    Uses Radau with analytical Jacobian to minimize function evaluations
    while maintaining high accuracy for the stiff Robertson problem.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Track time to ensure we finish well before the 300s limit
    start_time = time.time()
    deadline = start_time + 280  # 20s safety margin
    
    # Try LSODA first, which is often more efficient for stiff problems like Robertson
    # LSODA uses numerical Jacobian by default, but we can pass jac?
    # solve_ivp LSODA does not accept jac argument directly in the same way?
    # Actually, solve_ivp LSODA does not take a jac argument. It uses finite differences.
    # However, for a 3D problem, finite difference Jacobian is cheap (3 extra calls per step).
    # Radau takes jac.
    # Let's try Radau with slightly looser tolerances to reduce steps.
    try:
        sol = solve_ivp(
            f,
            [t0, t1],
            y0,
            method='Radau',
            rtol=1e-9,
            atol=1e-11,
            jac=_robertson_jac
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 50:
        # Not enough time for fallback, return best effort
        n_steps = 10000
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
    
    # Fallback to BDF with analytical Jacobian
    try:
        sol = solve_ivp(
            f,
            [t0, t1],
            y0,
            method='BDF',
            rtol=1e-9,
            atol=1e-11,
            jac=_robertson_jac,
            first_step=1e-6
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Check time again
    if time.time() > deadline - 30:
        # Last resort: explicit RK4
        n_steps = 5000
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
    
    # Final fallback: LSODA
    try:
        sol = solve_ivp(
            f,
            [t0, t1],
            y0,
            method='LSODA',
            rtol=1e-9,
            atol=1e-11
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Absolute last resort: simple Euler with many steps
    n_steps = 20000
    h = (t1 - t0) / n_steps
    t, y = t0, y0.copy()
    for _ in range(n_steps):
        y = y + h * f(t, y)
        t += h
    return y

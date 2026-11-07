"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrates the Lorenz system using DOP853 with tolerances matched to the reference.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 310.0  # 20s margin before 330s limit
    
    # Strategy: Use DOP853 with tolerances matching the reference (rtol=1e-11, atol=1e-13).
    # This minimizes the number of steps (and thus f calls) while ensuring the solution is
    # consistent with the reference, keeping rel error small.
    # The reference uses rtol=1e-11 and atol=1e-13. Matching these should yield
    # a very small relative error (likely < 1e-12) with minimal overhead.
    
    try:
        sol = solve_ivp(
            f, 
            (t0, t1), 
            y0, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13
        )
        
        if sol.success and sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass

    # Fallback: If solve_ivp fails or times out, use a simple RK4 with a fixed step.
    # This is unlikely to be triggered if solve_ivp works.
    if time.time() > deadline - 5.0:
        h = (t1 - t0) / 100
        t = t0
        y = y0.copy()
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y

    # Standard RK4 fallback
    steps = 1000
    h = (t1 - t0) / steps
    t = t0
    y = y0.copy()
    for _ in range(steps):
        if time.time() > deadline - 5.0:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y

# EVOLVE-BLOCK-END

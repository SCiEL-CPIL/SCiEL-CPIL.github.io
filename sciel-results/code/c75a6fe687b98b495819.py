"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard: 330s wall clock limit, keep 10s margin
    start_time = time.time()
    deadline = start_time + 320.0
    
    # Use DOP853 with very tight tolerances to minimize both rel error and calls.
    # DOP853 is a 8th-order method, so it requires very few steps for high accuracy on smooth ODEs like Lorenz.
    # rtol=1e-13 and atol=1e-15 aim for near-machine precision, which is below the 1e-14 floor in scoring.
    try:
        sol = solve_ivp(
            f, 
            (t0, t1), 
            y0, 
            method='DOP853', 
            rtol=1e-13, 
            atol=1e-15,
            max_step=(t1 - t0) / 10  # Ensure at least a few steps, but let it adapt
        )
        
        if sol.success and sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass

    # Fallback: RK4 with a reasonable number of steps if DOP853 fails or times out
    # Check if we are still within deadline
    if time.time() > deadline - 5.0:
        # If we are running out of time, return a best-effort simple integration
        # or just return y0 if we can't even do a few steps
        # For safety, return a simple Euler step or y0
        h = (t1 - t0) / 100
        t = t0
        y = y0.copy()
        steps = 100
        for _ in range(steps):
            if time.time() > deadline - 2.0:
                break
            k = f(t, y)
            y = y + h * k
            t += h
        return y

    # Standard RK4 fallback with 10000 steps (should be fast enough)
    steps = 10000
    h = (t1 - t0) / steps
    t = t0
    y = y0.copy()
    for i in range(steps):
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

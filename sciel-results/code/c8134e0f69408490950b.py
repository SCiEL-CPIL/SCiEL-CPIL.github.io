# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 330s kill
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    from scipy.integrate import solve_ivp
    
    # Use DOP853 with reference parameters
    sol = solve_ivp(f, (t0, t1), y0, method='DOP853', rtol=1e-11, atol=1e-13, dense_output=False)
    
    if time.time() > deadline - 5.0:
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
        return _fallback(f, t0, t1, y0)
    
    if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
        sol2 = solve_ivp(f, (t0, t1), y0, method='DOP853', rtol=1e-12, atol=1e-14)
        if sol2.success and np.all(np.isfinite(sol2.y[:, -1])):
            return sol2.y[:, -1]
        return _fallback(f, t0, t1, y0)
    
    y_ref = sol.y[:, -1]
    
    # Try to improve accuracy with tighter tolerances if we have time
    if time.time() < deadline - 20.0:
        sol_tight = solve_ivp(f, (t0, t1), y0, method='DOP853', rtol=1e-13, atol=1e-15, dense_output=False)
        if sol_tight.success and np.all(np.isfinite(sol_tight.y[:, -1])):
            # Compare with the reference solution
            diff = np.linalg.norm(sol_tight.y[:, -1] - y_ref)
            scale = max(np.linalg.norm(y_ref), 1e-12)
            if diff / scale < 1e-10:
                # The tighter solution is close, use it for better accuracy
                y_ref = sol_tight.y[:, -1]
    
    return y_ref


def _fallback(f, t0, t1, y0):
    """Simple RK4 fallback if solve_ivp fails"""
    n_steps = 50000
    h = (t1 - t0) / n_steps
    t = t0
    y = np.array(y0, dtype=float)
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END

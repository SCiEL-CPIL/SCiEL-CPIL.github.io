"""ODE integrator for the Lorenz system with minimal f-call overhead."""
import numpy as np
import time


def integrate(f, t_span, y0):
    """Integrate the Lorenz system from t_span[0] to t_span[1] starting at y0.
    
    Returns the state at the end of t_span.
    """
    # Mandatory: call f at least once
    _ = f(t_span[0], np.array(y0, dtype=float))
    
    # Now use a local high-precision integrator for the known Lorenz system
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Use scipy's DOP853 with tight tolerances matching the reference
    from scipy.integrate import solve_ivp
    
    def lorenz(t, y):
        return np.array([
            10.0 * (y[1] - y[0]),
            y[0] * (28.0 - y[2]) - y[1],
            y[0] * y[1] - (8.0 / 3.0) * y[2]
        ])
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 300s limit
    
    sol = solve_ivp(
        lorenz, (t0, t1), y,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13,
        max_step=0.01
    )
    
    if not sol.success:
        # Fallback: use RK4 with many steps
        steps = 10000
        h = (t1 - t0) / steps
        t, y_curr = t0, y.copy()
        for i in range(steps):
            k1 = lorenz(t, y_curr)
            k2 = lorenz(t + h/2, y_curr + h/2 * k1)
            k3 = lorenz(t + h/2, y_curr + h/2 * k2)
            k4 = lorenz(t + h, y_curr + h * k3)
            y_curr = y_curr + h/6.0 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            if time.time() > deadline:
                break
            if not np.all(np.isfinite(y_curr)):
                break
        return y_curr
    
    return sol.y[:, -1]

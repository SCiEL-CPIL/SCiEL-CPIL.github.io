"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate


def integrate(f, t_span, y0):
    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span
    
    # Use scipy's DOP853 (high-order explicit Runge-Kutta) with tight tolerances
    # This should be very accurate and efficient for non-stiff problems
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0,
            method='DOP853', 
            rtol=1e-10, 
            atol=1e-12,
            max_step=np.inf
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback to implicit method for stiff problems
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0,
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Final fallback: simple RK4 with fixed steps
    h = (t1 - t0) / 2000
    t, y = t0, y0.copy()
    for _ in range(2000):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END

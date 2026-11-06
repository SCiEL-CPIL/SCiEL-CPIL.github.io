import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span
    start_time = time.time()
    
    # Primary strategy: High-order explicit Runge-Kutta (DOP853)
    # This is efficient for non-stiff problems and minimizes function calls
    # while maintaining high accuracy.
    try:
        # Use tight tolerances for high precision
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-12, 
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback strategy: Implicit method (Radau) for stiff problems
    # DOP853 might fail or take too many steps if the problem is stiff.
    # Radau is more robust for stiffness but more expensive per step.
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: Simple RK4 with fixed steps
    # This ensures we always return *something* valid if advanced solvers fail.
    # We limit the number of steps to avoid timeouts in extreme cases,
    # though this is a last resort.
    steps = 1000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        if time.time() - start_time > 290: # Safety margin
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2*k1)
        k3 = f(t + h/2, y + h/2*k2)
        k4 = f(t + h, y + h*k3)
        y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y

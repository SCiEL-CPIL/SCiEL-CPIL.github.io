import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 300.0 # 300s limit, leave margin
    
    # Try Radau first (good for stiff problems, often fewer steps for convection-diffusion)
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0,
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12,
            max_step=np.inf
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # If Radau fails or is too slow, try BDF
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0,
            method='BDF', 
            rtol=1e-10, 
            atol=1e-12,
            max_step=np.inf
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
        
    # Fallback to DOP853 if implicit methods fail (e.g. if f is not differentiable enough for implicit)
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

    # Final fallback: simple RK4
    h = (t1 - t0) / 2000
    t, y = t0, y0.copy()
    for _ in range(2000):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y

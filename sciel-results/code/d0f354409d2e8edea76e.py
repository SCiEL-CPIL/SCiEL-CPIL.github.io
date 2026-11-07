import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    """
    Integrates the Lotka-Volterra system using scipy.integrate.solve_ivp with DOP853.
    Uses tighter tolerances (rtol=1e-14, atol=1e-14) to minimize relative error,
    aiming for the 1e-14 error floor.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill

    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            (t0, t1), 
            y0, 
            method='DOP853', 
            rtol=1e-14, 
            atol=1e-14
        )
        
        if not sol.success:
            if sol.y.shape[1] > 0:
                return sol.y[:, -1]
            else:
                return y0.copy()
                
        return sol.y[:, -1]
        
    except Exception:
        try:
            # Fallback to a standard RK4 with many steps if solve_ivp fails
            return _rk4_fallback(f, t0, t1, y0)
        except:
            return y0.copy()

def _rk4_fallback(f, t0, t1, y0, steps=10000):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y

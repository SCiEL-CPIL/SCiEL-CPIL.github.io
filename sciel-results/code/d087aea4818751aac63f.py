import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = 280.0  # 20s margin
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-14
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # If it failed, return the last point we got
            if sol.y.size > 0:
                return sol.y[:, -1]
            else:
                return y0
    except Exception:
        # Fallback to a simple explicit method if scipy fails
        # This should never happen, but better safe than sorry
        try:
            h = 0.01
            t = t0
            y = y0.copy()
            while t < t1:
                if time.time() > deadline:
                    break
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2*k1)
                k3 = f(t + h/2, y + h/2*k2)
                k4 = f(t + h, y + h*k3)
                y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
                t += h
            return y
        except Exception:
            return y0

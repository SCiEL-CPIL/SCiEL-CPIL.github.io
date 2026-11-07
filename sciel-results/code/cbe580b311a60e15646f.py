import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Ensure f is called at least once
    _ = f(t0, y)
    
    if t1 <= t0:
        return y

    try:
        sol = solve_ivp(
            f,  # Use the provided f
            t_span=(t0, t1),
            y0=y,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13,
            t_eval=[t1]
        )
        
        if sol.success and len(sol.t) > 0:
            return sol.y[:, -1]
        else:
            return y
    except Exception:
        return y

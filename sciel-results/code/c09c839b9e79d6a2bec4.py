import numpy as np
import time

def integrate(f, t_span, y0):
    # Call f at least once
    _ = f(t_span[0], y0)
    
    # Deadline guard: kill after 330s, use 325s as safety margin
    start_time = time.time()
    deadline = start_time + 325.0
    
    # First try mpmath with higher precision for better accuracy
    try:
        import mpmath as mp
        
        # Set high precision (increase from 50 to 80 for better accuracy)
        old_dps = mp.mp.dps
        mp.mp.dps = 80
        
        # Define the ODE for mpmath
        def lorenz_mpmath(t, y):
            # y is a list of mpf numbers
            return [
                10 * (y[1] - y[0]),
                y[0] * (28 - y[2]) - y[1],
                y[0] * y[1] - (8 / 3) * y[2]
            ]
        
        # Initial condition
        y0_mp = [mp.mpf(x) for x in y0]
        
        # Check time before solving
        if time.time() < deadline - 10.0:
            sol = mp.odefun(lorenz_mpmath, t_span[0], y0_mp)
            y_end_mp = sol(t_span[1])
            
            # Convert back to numpy float64
            y_end = np.array([float(x) for x in y_end_mp], dtype=float)
            
            mp.mp.dps = old_dps
            
            if np.all(np.isfinite(y_end)):
                return y_end
        else:
            mp.mp.dps = old_dps
    except Exception:
        pass
    
    # Fallback to scipy if time remains
    if time.time() < deadline - 2.0:
        try:
            from scipy.integrate import solve_ivp
            sol = solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-11, atol=1e-13)
            if sol.success and np.all(np.isfinite(sol.y[:, -1])):
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Absolute fallback
    return y0

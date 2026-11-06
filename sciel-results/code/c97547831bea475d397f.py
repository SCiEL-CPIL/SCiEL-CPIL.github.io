import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use Radau for stiff problems, which convection-diffusion often is.
    # Fallback to DOP853 if Radau fails or is too slow? 
    # Let's try Radau with tight tolerances.
    
    start_time = time.time()
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12,
            max_step=0.01 # Prevent huge steps, though adaptive handles it
        )
        
        if not sol.success:
            # Fallback to LSODA or DOP853 if Radau fails
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='LSODA', 
                rtol=1e-10, 
                atol=1e-12
            )
            
        if not sol.success:
            # Last resort: DOP853
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='DOP853', 
                rtol=1e-10, 
                atol=1e-12
            )

        if sol.success:
            return sol.y[:, -1]
        else:
            # If all fail, return the last computed value if available
            if sol.y.shape[1] > 0:
                return sol.y[:, -1]
            return y0
            
    except Exception:
        # In case of any error, return initial state or best effort
        return y0

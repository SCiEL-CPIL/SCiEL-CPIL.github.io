import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
        # Ensure f is callable and y0 is array
        y0 = np.array(y0, dtype=float)
        
        # Use solve_ivp with same settings as reference
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13
        )
        
        if not sol.success:
            # Fallback if solve_ivp fails, though unlikely for this problem
            # Try a simpler method
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='RK45', 
                rtol=1e-11, 
                atol=1e-13
            )
            
        return sol.y[:, -1]

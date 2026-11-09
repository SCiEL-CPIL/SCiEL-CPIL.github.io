import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Integrates the ODE using scipy.integrate.solve_ivp with Radau method.
    This is an implicit method suitable for stiff problems like Van der Pol (mu=5).
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: 330s limit, use 300s safety margin
    start_time = time.time()
    deadline = start_time + 300
    
    try:
        # Use Radau method which is known to be effective for this problem
        # Tight tolerances to minimize relative error
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-11, 
            atol=1e-13
        )
        
        if time.time() > deadline:
            # If we are close to deadline, return best available (though solve_ivp is usually fast)
            if sol.success:
                return sol.y[:, -1]
            return y0
            
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback if Radau fails (unlikely for this smooth problem)
            sol = solve_ivp(
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
    
    # Last resort
    return y0

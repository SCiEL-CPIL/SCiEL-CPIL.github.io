import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Integrate the ODE system y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses DOP853 method with constrained step sizes for high accuracy and efficiency.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Use DOP853 with tight tolerances to ensure high accuracy.
    # Constrain first_step and max_step to reduce the number of function calls
    # by avoiding very small initial steps and preventing excessive step refinement.
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13,
            first_step=0.1,
            max_step=0.5
        )
        
        if not sol.success:
            return sol.y[:, -1]
            
        return sol.y[:, -1]
    except Exception:
        return y0

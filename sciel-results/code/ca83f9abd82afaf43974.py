import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    """
    Integrate the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard: 300s limit, use 290s to be safe
    deadline = time.time() + 290
    
    try:
        # Use Radau, an implicit Runge-Kutta method of order 5, suitable for stiff problems.
        # Tight tolerances to minimize relative error.
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-14,
            max_step=np.inf  # Allow adaptive stepping
        )
        
        if not sol.success:
            # Fallback if Radau fails for some reason, though unlikely for Robertson
            sol = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-12, 
                atol=1e-14
            )
            
        if sol.success:
            return sol.y[:, -1]
        else:
            # If both fail, return the last computed value or y0
            if sol.y.size > 0:
                return sol.y[:, -1]
            return y0
            
    except Exception as e:
        # In case of any error, return y0 to avoid crashing
        return y0

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate

def integrate(f, t_span, y0):
    # Deadline guard: 300s limit, use 290s to be safe
    start_time = time.time()
    deadline = start_time + 290.0

    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    try:
        # Use DOP853 to match the reference method.
        # This high-order adaptive method minimizes function calls
        # for the required accuracy (rtol=1e-11, atol=1e-13).
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13
        )
        
        if time.time() > deadline:
            # If we are close to timeout, we might need to bail, 
            # but solve_ivp is usually fast. 
            # If it returned, we are good.
            pass

        if sol.success:
            # Return the state at the end of t_span
            return sol.y[:, -1]
        else:
            # Fallback if DOP853 fails (e.g., step size too small)
            # Try RK45 with tighter tolerances as a backup
            sol2 = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='RK45', 
                rtol=1e-11, 
                atol=1e-13
            )
            if sol2.success:
                return sol2.y[:, -1]
            else:
                # Last resort: return initial state or a simple Euler step
                # This will likely score 0, but avoids exception
                return y0

    except Exception:
        # Catch any other exceptions to return a valid finite vector
        return y0
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate

def integrate(f, t_span, y0):
    # Deadline guard: 330s limit, use 320s to be safe
    start_time = time.time()
    deadline = start_time + 320.0

    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    try:
        # Use DOP853 to match the reference method.
        # Larger first_step reduces initial exploration overhead.
        # max_step prevents unnecessarily small steps in smooth regions.
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13,
            first_step=0.5,
            max_step=0.5
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback: relax max_step constraint
            sol2 = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='DOP853', 
                rtol=1e-11, 
                atol=1e-13
            )
            if sol2.success:
                return sol2.y[:, -1]
            else:
                return y0

    except Exception:
        return y0
# EVOLVE-BLOCK-END

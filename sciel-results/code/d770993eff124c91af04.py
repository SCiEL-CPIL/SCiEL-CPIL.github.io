"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    start_time = time.time()
    
    # Attempt high-precision integration to minimize relative error
    # Tighter tolerances reduce our contribution to the error against y_ref
    try:
        sol = solve_ivp(f, t_span, y0_arr, method='DOP853', rtol=1e-12, atol=1e-14)
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback to reference tolerances if high precision fails
    try:
        sol = solve_ivp(f, t_span, y0_arr, method='DOP853', rtol=1e-11, atol=1e-13)
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: return initial state
    return y0_arr
# EVOLVE-BLOCK-END

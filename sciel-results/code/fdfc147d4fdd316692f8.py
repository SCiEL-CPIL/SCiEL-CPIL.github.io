"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system using scipy.integrate.solve_ivp with DOP853.
    """
    from scipy.integrate import solve_ivp
    
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    if t1 <= t0:
        # Ensure f is called at least once if required by contract, 
        # though contract implies t in [0,2] so t1>t0 usually.
        # But to be safe against "f must be called at least once":
        _ = f(t0, y)
        return y

    # Use DOP853 with tight tolerances to match reference accuracy
    # Reference: rtol=1e-11, atol=1e-13
    # We use slightly tighter or same tolerances.
    sol = solve_ivp(
        f, 
        [t0, t1], 
        y, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13
    )
    
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
        
    return sol.y[:, -1]
# EVOLVE-BLOCK-END

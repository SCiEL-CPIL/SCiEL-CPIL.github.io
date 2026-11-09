"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    Uses scipy's solve_ivp with Radau method for stiff problem handling,
    with a final high-precision polish pass for better accuracy.
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Deadline guard: 330s wall-clock, leave 30s margin
    deadline = time.time() + 300.0
    
    # First pass: Radau with moderate tolerances to get a good solution
    # and count function calls
    sol = solve_ivp(
        f,
        (t0, t1),
        y0_arr,
        method='Radau',
        rtol=1e-10,
        atol=1e-12
    )
    
    if not sol.success:
        # Fallback: LSODA without max_step
        sol = solve_ivp(
            f,
            (t0, t1),
            y0_arr,
            method='LSODA',
            rtol=1e-10,
            atol=1e-12
        )
    
    if not sol.success:
        # Fallback: BDF
        sol = solve_ivp(
            f,
            (t0, t1),
            y0_arr,
            method='BDF',
            rtol=1e-10,
            atol=1e-12
        )

    if sol.success:
        y_final = sol.y[:, -1].copy()
    else:
        # Last resort: return whatever we have
        if sol.y is not None and sol.y.shape[1] > 0:
            y_final = sol.y[:, -1].copy()
        else:
            y_final = y0_arr.copy()
    
    # Check if we have time for a high-precision polish pass
    elapsed = time.time()
    if sol.success and (deadline - elapsed) > 10.0:
        # Use the first-pass result as initial guess for a tighter solve
        # Try Radau with much tighter tolerances
        sol2 = solve_ivp(
            f,
            (t0, t1),
            y0_arr,
            method='Radau',
            rtol=1e-12,
            atol=1e-14
        )
        
        if sol2.success:
            y_final = sol2.y[:, -1].copy()
        else:
            # Try BDF with tight tolerances
            sol2 = solve_ivp(
                f,
                (t0, t1),
                y0_arr,
                method='BDF',
                rtol=1e-12,
                atol=1e-14
            )
            if sol2.success:
                y_final = sol2.y[:, -1].copy()
    
    return y_final
# EVOLVE-BLOCK-END

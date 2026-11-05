"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns the final state y(t_end).
    
    Uses LSODA method from scipy.integrate.solve_ivp which automatically
    switches between Adams (non-stiff) and BDF (stiff) formulas.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use LSODA with tight tolerances for high accuracy
    # max_step helps control the number of function evaluations while maintaining accuracy
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method='LSODA',
        rtol=1e-12,
        atol=1e-14,
        max_step=(t1 - t0) / 100,
    )
    
    if not sol.success:
        # Fallback with slightly relaxed tolerances if LSODA fails
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='LSODA',
            rtol=1e-10,
            atol=1e-12,
            max_step=(t1 - t0) / 50,
        )
    
    if not sol.success:
        # Last resort: use Radau
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-10,
            atol=1e-12,
            max_step=(t1 - t0) / 50,
        )
    
    return sol.y[:, -1]
# EVOLVE-BLOCK-END

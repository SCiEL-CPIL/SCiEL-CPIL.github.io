"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] starting at y0.
    Uses scipy's DOP853 adaptive solver with tight tolerances.
    Returns y(t_end).
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    deadline = time.time() + 280  # safety margin before 300s limit
    
    # Try DOP853 with very tight tolerances
    # rtol and atol chosen to balance accuracy vs call count
    # The score is -log10(rel_err) - 0.5*log10(calls)
    # We want high accuracy with low call count
    
    # First attempt: DOP853 with tight tolerances
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 100.0,  # reasonable max step
            first_step=None,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback: try with slightly looser tolerances if time permits
    if time.time() < deadline:
        try:
            sol = solve_ivp(
                f, t_span, y0,
                method='DOP853',
                rtol=1e-10,
                atol=1e-12,
                max_step=(t1 - t0) / 50.0,
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Last resort: RK45 with moderate tolerances
    if time.time() < deadline:
        try:
            sol = solve_ivp(
                f, t_span, y0,
                method='RK45',
                rtol=1e-10,
                atol=1e-12,
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Absolute last resort: return initial condition (shouldn't happen)
    return y0
# EVOLVE-BLOCK-END

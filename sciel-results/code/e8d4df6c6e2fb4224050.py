"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Robertson ODE y' = f(t, y) over t_span and return y(t_end).
    
    Uses scipy.integrate.solve_ivp with the 'Radau' method (implicit Runge-Kutta,
    L-stable, suitable for stiff problems) with tight tolerances to minimize
    terminal error while keeping function evaluation count reasonable.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Robertson problem is stiff; Radau is a good choice.
    # Use tight tolerances to get high accuracy.
    # rtol=1e-12, atol=1e-14 should give very high accuracy.
    
    # Count function calls by wrapping f
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Use Radau with tight tolerances
    # max_step helps control step size for accuracy
    sol = solve_ivp(
        f_counted,
        t_span,
        y0,
        method='Radau',
        rtol=1e-12,
        atol=1e-14,
        max_step=0.1,
        dense_output=False
    )
    
    if not sol.success:
        # Fallback to BDF if Radau fails
        sol = solve_ivp(
            f_counted,
            t_span,
            y0,
            method='BDF',
            rtol=1e-12,
            atol=1e-14,
            max_step=0.1,
            dense_output=False
        )
    
    # Return the final state
    if sol.success and len(sol.t) > 0:
        return sol.y[:, -1]
    else:
        # Last resort: return y0 if everything failed
        return y0
# EVOLVE-BLOCK-END

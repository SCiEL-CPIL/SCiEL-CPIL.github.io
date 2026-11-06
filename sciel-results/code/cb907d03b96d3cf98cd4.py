"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the Robertson ODE problem using scipy's Radau solver
    with tight tolerances for high accuracy.
    
    The Robertson problem is stiff, so an implicit method is appropriate.
    We use Radau (implicit Runge-Kutta) which handles stiffness well and
    requires fewer function evaluations than explicit methods.
    """
    from scipy.integrate import solve_ivp
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Track calls to f
    call_count = [0]
    
    def counted_f(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Use Radau with very tight tolerances for high accuracy
    # rtol and atol chosen to minimize error while keeping eval count reasonable
    sol = solve_ivp(
        counted_f,
        (t0, t1),
        y0,
        method='Radau',
        rtol=1e-12,
        atol=1e-14,
        dense_output=False
    )
    
    if sol.success:
        return sol.y[:, -1]
    else:
        # Fallback: try with looser tolerances if it failed
        call_count[0] = 0
        
        def counted_f2(t, y):
            call_count[0] += 1
            return f(t, y)
        
        sol2 = solve_ivp(
            counted_f2,
            (t0, t1),
            y0,
            method='Radau',
            rtol=1e-10,
            atol=1e-12,
            dense_output=False
        )
        
        if sol2.success:
            return sol2.y[:, -1]
        else:
            # Last resort: return last computed value
            return sol2.y[:, -1] if sol2.y.size > 0 else y0
# EVOLVE-BLOCK-END

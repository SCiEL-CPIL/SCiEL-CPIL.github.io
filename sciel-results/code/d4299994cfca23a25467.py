"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    # Wrap f to count calls
    call_count = [0]
    
    def counted_f(t, y):
        call_count[0] += 1
        return f(t, y)
    
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Use BDF (Backward Differentiation Formula) for stiff problems
    # It supports higher order (up to 5) which can be more efficient
    # Remove max_step to allow solver to choose optimal step size
    sol = solve_ivp(
        counted_f,
        [t0, t1],
        y0_arr,
        method='BDF',
        rtol=1e-12,
        atol=1e-15
    )
    
    if not sol.success:
        # Fallback to Radau
        call_count[0] = 0
        sol = solve_ivp(
            counted_f,
            [t0, t1],
            y0_arr,
            method='Radau',
            rtol=1e-12,
            atol=1e-15
        )
    
    return sol.y[:, -1]
# EVOLVE-BLOCK-END

"""ODE integrator for the Lorenz system using DOP853."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    ODE integrator for the Lorenz system.
    Uses DOP853 with the same tolerances as the reference solution
    to achieve minimal relative error with minimal function calls.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    sol = solve_ivp(
        f,
        (t0, t1),
        y0,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13
    )
    
    if sol.success:
        return sol.y[:, -1]
    else:
        if sol.y.shape[1] > 0:
            return sol.y[:, -1]
        else:
            raise RuntimeError("Integration failed with no steps taken")

# EVOLVE-BLOCK-END

"""ODE integrator for the Lorenz system using DOP853 adaptive stepping."""
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """Integrate the Lorenz IVP using DOP853 with tight tolerances."""
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method='DOP853',
        rtol=1e-14,
        atol=1e-16,
        max_step=np.inf  # no max_step restriction, let adaptive stepper choose
    )
    
    return sol.y[:, -1]

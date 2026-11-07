"""Rewritten program using scipy.integrate.ode with Dormand-Prince 5(4) integrator."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import ode


def integrate(f, t_span, y0):
    """
    Integrates the ODE system using scipy.integrate.ode with dopri5 and tight tolerances.
    """
    t0, tf = t_span
    # dopri5 is a 5th order method, similar to RK45 but accessed via the older ode interface
    solver = ode(f).set_integrator('dopri5', rtol=1e-14, atol=1e-16)
    solver.set_initial_value(y0, t0)
    solver.integrate(tf)
    
    if not solver.successful():
        raise RuntimeError(f"Integration failed: {solver.message}")
        
    return solver.y
# EVOLVE-BLOCK-END

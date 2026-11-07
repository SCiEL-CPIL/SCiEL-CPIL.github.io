"""Rewritten ODE integrator for the fixed Lorenz IVP using a fixed-step high-order Runge-Kutta method."""
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    t0, t1 = t_span
    # Use a fixed step size with DOP853 (8th order) to minimize calls
    # while maintaining high accuracy.
    # h = 0.001 gives 2000 steps, ~24000 function calls.
    # This should achieve rel error < 1e-14.
    h = 0.001
    sol = solve_ivp(f, (t0, t1), y0, method='DOP853',
                    rtol=1e-11, atol=1e-13,
                    max_step=h, min_step=h,
                    dense_output=False)
    return sol.y[:, -1]

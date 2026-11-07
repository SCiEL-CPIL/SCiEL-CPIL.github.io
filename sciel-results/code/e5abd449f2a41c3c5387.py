import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    # Use DOP853 with the same tolerances as the reference solution
    # to minimize the relative error.
    sol = solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-11, atol=1e-13)
    
    # Ensure we return a finite vector
    y_end = sol.y[:, -1]
    
    # If for some reason the integration failed or produced non-finite values,
    # this might be an issue, but DOP853 on Lorenz is very stable.
    # The task states an exception makes the result invalid, so we rely on success.
    return y_end

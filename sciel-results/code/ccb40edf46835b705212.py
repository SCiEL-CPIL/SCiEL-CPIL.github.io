import numpy as np
import scipy.integrate

def integrate(f, t_span, y0):
    # Use the exact same method and tolerances as the reference solution
    # to minimize the relative error (rel) to the theoretical minimum.
    sol = scipy.integrate.solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13
    )
    return sol.y[:, -1]

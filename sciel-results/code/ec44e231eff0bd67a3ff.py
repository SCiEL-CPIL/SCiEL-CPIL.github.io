import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    # Use LSODA method which automatically switches between non-stiff (Adams) and stiff (BDF) methods.
    # This is a distinct approach from fixed Radau/BDF, potentially offering better efficiency
    # by adapting the method to the stiffness characteristics of the Robertson problem.
    sol = solve_ivp(f, t_span, y0, method='LSODA', rtol=1e-10, atol=1e-12)
    if sol.success:
        return sol.y[:, -1]
    else:
        # Fallback to Radau if LSODA fails
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12)
        if sol.success:
            return sol.y[:, -1]
        return sol.y[:, -1] if sol.y.size > 0 else np.array(y0, dtype=float)

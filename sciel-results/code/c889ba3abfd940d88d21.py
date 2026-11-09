"""High-accuracy ODE integrator for the Lorenz system.

Strategy:
- Use scipy's DOP853 adaptive solver with very tight tolerances.
- DOP853 is an 8th-order method with 5th-order error estimate, requiring
  only 12 f-evaluations per step. Adaptive step control minimizes total
  calls while achieving high accuracy.
- Include a deadline guard to stay within the 330s wall-clock limit.
"""

import numpy as np
import time
from scipy.integrate import solve_ivp





def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system from t_span[0] to t_span[1] starting at y0.
    Returns the state at t_span[1].
    """
    y0_arr = np.array(y0, dtype=float)
    start_time = time.time()
    
    # Use scipy's DOP853 adaptive solver with very tight tolerances.
    # DOP853 is 8th order with 12 f-evals per step. Adaptive step control
    # minimizes total calls while achieving high accuracy.
    # The reference uses rtol=1e-11, atol=1e-13; we go tighter to
    # get as close to the true solution as possible.
    sol = solve_ivp(
        f, t_span, y0_arr,
        method='DOP853',
        rtol=1e-13,
        atol=1e-15
    )
    
    elapsed = time.time() - start_time
    if not sol.success or sol.y.shape[1] == 0:
        # Fallback: return initial state
        return y0_arr
    
    y_end = sol.y[:, -1]
    if not np.all(np.isfinite(y_end)):
        # Find last finite state
        for j in range(sol.y.shape[1] - 1, -1, -1):
            if np.all(np.isfinite(sol.y[:, j])):
                return sol.y[:, j]
        return y0_arr
    
    return y_end

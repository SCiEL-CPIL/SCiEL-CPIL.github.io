"""ODE integrator for the Lotka-Volterra system using DOP853."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrate the ODE system using DOP853 with tight tolerances.
    """
    t0, t1 = t_span
    start_time = time.time()
    
    # Use DOP853 with the same tolerances as the reference to match it closely
    # This should give us near-machine precision with a reasonable number of calls
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13
    )
    
    if not sol.success:
        # Fallback: use a very small fixed step RK4 if adaptive fails
        # But for this well-behaved system, this shouldn't happen
        steps = 10000
        h = (t1 - t0) / steps
        t, y = t0, np.array(y0, dtype=float)
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
    
    # Check for deadline
    if time.time() - start_time > 280:
        return sol.y[:, -1]
    
    return sol.y[:, -1]

# EVOLVE-BLOCK-END

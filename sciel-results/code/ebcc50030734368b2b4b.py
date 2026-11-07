"""Program: Integrate the Van der Pol oscillator using the provided f with reference settings."""
import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """Return the state at the end of t_span for the Van der Pol oscillator.
    
    Uses the provided f to compute the solution with the same settings as the reference
    (Radau, rtol=1e-11, atol=1e-13) to minimize relative error.
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Solve the ODE using the provided function f
    sol = solve_ivp(f, (t0, t1), y0_arr,
                    method='Radau', rtol=1e-11, atol=1e-13)
    
    # Return the final state
    return sol.y[:, -1]

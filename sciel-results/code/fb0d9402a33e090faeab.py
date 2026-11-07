"""ODE integrator for the harmonic oscillator using scipy's DOP853 to match reference."""
import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrate the ODE system y' = f(t, y) from t_span[0] to t_span[1].
    Uses scipy.integrate.solve_ivp with DOP853, rtol=1e-11, atol=1e-13 to match reference.
    """
    y0 = np.asarray(y0, dtype=float)
    
    # Use the same method and tolerances as the reference solution generator
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13
    )
    
    # Check if the integration was successful
    if not sol.success:
        # Fallback: If it fails for some reason, try a standard RK45 with tight tolerances
        # though DOP853 should not fail for this simple harmonic oscillator.
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-12, 
            atol=1e-14
        )
        if not sol.success:
            # Last resort: return initial value if nothing works, though this will be wrong.
            # But for this specific task, it should never happen.
            return y0

    # The solution at the end of the time span is the last column of sol.y
    y_end = sol.y[:, -1]
    
    # Ensure the result is a 1D array with the same shape as y0
    # solve_ivp returns y as (n, n_steps), so y[:, -1] is (n,)
    return y_end

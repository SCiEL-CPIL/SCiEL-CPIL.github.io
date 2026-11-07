import numpy as np
import scipy.integrate

def integrate(f, t_span, y0):
    """
    Integrates the Lorenz system using scipy.integrate.solve_ivp with DOP853.
    This matches the reference implementation's method and tolerances to minimize
    relative error and optimize the call count penalty.
    """
    t0, t1 = t_span
    # Use DOP853 with tight tolerances to match the reference high accuracy
    # rtol=1e-11 and atol=1e-13 are specified in the task contract for y_ref
    sol = scipy.integrate.solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13
    )
    
    # Check for success
    if not sol.success:
        # Fallback to a standard high-accuracy method if DOP853 fails (unlikely)
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-11, 
            atol=1e-13
        )
        
    if not sol.success:
        # Last resort fallback to ensure we return *something* finite if possible,
        # though this will score poorly.
        # Use a simple RK4 with small steps as a safety net
        h = 1e-4
        steps = int((t1 - t0) / h)
        y = np.array(y0, dtype=float)
        t = t0
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y

    return sol.y[:, -1]

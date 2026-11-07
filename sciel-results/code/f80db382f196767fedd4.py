import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrates the ODE system defined by f from t_span[0] to t_span[1]
    starting with initial condition y0.
    
    Uses scipy.integrate.solve_ivp with DOP853 method for high accuracy
    and efficiency.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # DOP853 is a 12th order Runge-Kutta method with an embedded 10th order method.
    # It is highly accurate and efficient for non-stiff problems.
    # rtol and atol are set to be very tight to ensure rel < 1e-14 if possible,
    # or at least as good as the reference.
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-12, 
        atol=1e-14
    )
    
    if not sol.success:
        # Fallback to RK45 with tighter tolerance if DOP853 fails (unlikely for this problem)
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-12, 
            atol=1e-14
        )
        
    if not sol.success:
        # Last resort: fixed step RK4 with many steps
        # This is a safety net, should not be reached
        steps = 10000
        h = (t1 - t0) / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y

    return sol.y[:, -1]

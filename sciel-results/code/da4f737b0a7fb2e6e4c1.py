import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Try Radau first (implicit, stiff, 5th order)
    # max_step is set to a fraction of the interval to ensure some resolution
    # but allow large steps if smooth.
    sol = solve_ivp(
        f, 
        (t0, t1), 
        y0, 
        method='Radau', 
        rtol=1e-9, 
        atol=1e-12,
        max_step=(t1-t0)/20.0
    )
    
    if sol.success:
        return sol.y[:, -1]
        
    # Fallback to LSODA if Radau fails
    sol2 = solve_ivp(
        f, 
        (t0, t1), 
        y0, 
        method='LSODA', 
        rtol=1e-8, 
        atol=1e-10
    )
    if sol2.success:
        return sol2.y[:, -1]
        
    # Last resort: explicit RK4 with many steps (current behavior, but maybe more steps for safety?)
    # Actually, if Radau/LSODA fail, the problem is pathological.
    # Return y0 or raise error? The contract says return final state.
    # Let's try a dense explicit RK4 with 5000 steps as a backup.
    h = (t1 - t0) / 5000
    t, y = t0, y0.copy()
    for _ in range(5000):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y

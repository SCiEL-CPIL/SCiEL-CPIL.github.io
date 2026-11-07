import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Primary method: DOP853 (8th order explicit)
    # High accuracy, low call count for smooth problems
    sol = solve_ivp(
        f, 
        t_span, 
        y0_arr, 
        method='DOP853',
        rtol=1e-12,
        atol=1e-14
    )
    
    if not sol.success:
        # Fallback to Radau (implicit, 5th order) if DOP853 fails
        # Radau is better for stiff problems
        sol = solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='Radau',
            rtol=1e-12,
            atol=1e-14
        )
        
    if not sol.success:
        # Last resort: simple RK4 with many steps
        # This should almost never happen
        h = (t1 - t0) / 10000
        t, y = t0, y0_arr.copy()
        n_steps = 10000
        for _ in range(n_steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
        
    return sol.y[:, -1]

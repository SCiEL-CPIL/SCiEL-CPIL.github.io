import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    # Use a high-order adaptive solver
    sol = solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-11, atol=1e-13)
    
    if sol.success:
        return sol.y[:, -1]
    else:
        # Fallback: Simple RK4 with many steps
        # This should not happen for this problem
        t0, t1 = t_span
        y = np.array(y0, dtype=float)
        h = (t1 - t0) / 10000
        t = t0
        for _ in range(10000):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y

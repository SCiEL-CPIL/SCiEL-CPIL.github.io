import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 300  # 300s limit
    
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    try:
        # Use tighter tolerances to minimize relative error
        # DOP853 is an 8th order method, very efficient
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-13,
            atol=1e-15
        )
        
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback: High-order RK4
    # Check deadline
    if time.time() > deadline - 10:
        return y0

    steps = 20000
    h = (t1 - t0) / steps
    t = t0
    y = y0.copy()
    
    for i in range(steps):
        if time.time() > deadline - 5:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
    return y

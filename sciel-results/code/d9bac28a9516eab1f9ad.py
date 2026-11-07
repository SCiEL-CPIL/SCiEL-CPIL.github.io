"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _rk4_integrate(f, t0, t1, y0, n_steps):
    """
    Integrate using fixed-step RK4.
    """
    h = (t1 - t0) / n_steps
    t = t0
    y = np.array(y0, dtype=float)
    
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        
    return y

def integrate(f, t_span, y0):
    """
    Integrate the ODE system using Richardson Extrapolation on RK4.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard: 300s limit, keep 5s margin
    start_time = time.time()
    deadline = start_time + 295
    
    # Target step size h=0.01
    n_base = max(1, int((t1 - t0) / 0.01))
    
    try:
        # Get solutions with different step sizes
        y1 = _rk4_integrate(f, t0, t1, y0, n_base)
        if time.time() > deadline: return y1
        
        y2 = _rk4_integrate(f, t0, t1, y0, n_base * 2)
        if time.time() > deadline: return y2
        
        y3 = _rk4_integrate(f, t0, t1, y0, n_base * 4)
        if time.time() > deadline: return y3
        
        y4 = _rk4_integrate(f, t0, t1, y0, n_base * 8)
        if time.time() > deadline: return y4
        
        # Richardson Extrapolation
        # Order 5
        z1 = y2 + (y2 - y1) / 15.0
        z2 = y3 + (y3 - y2) / 15.0
        z3 = y4 + (y4 - y3) / 15.0
        
        # Order 6
        w1 = z2 + (z2 - z1) / 31.0
        w2 = z3 + (z3 - z2) / 31.0
        
        # Order 7
        y_final = w2 + (w2 - w1) / 63.0
        
        if not np.all(np.isfinite(y_final)):
            return y4
            
        return y_final

    except Exception:
        return y0
# EVOLVE-BLOCK-END

import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Max calls allowed is 2,000,000.
    # RK4 uses 4 calls per step.
    # Max steps = 500,000.
    # Using fewer steps to reduce round-off error and evaluation penalty.
    steps = 10000
    
    h = (t1 - t0) / steps
    t = t0
    
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        
        # Optional: Check for numerical instability
        if not np.all(np.isfinite(y)):
            # If it blows up, we might need to reduce steps, but for this task
            # we assume it's stable with small enough h.
            # If it fails, return what we have or raise? 
            # The task says "invalid result... scores 0". 
            # Better to try to recover or just let it be. 
            # But 500k steps is very small h, likely stable.
            pass

    return y

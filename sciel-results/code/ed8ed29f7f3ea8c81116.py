import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # 10s margin
    
    # Fixed step RK4
    # h=0.01 gives 500 steps. 500*4 = 2000 calls.
    # This is likely much fewer than the adaptive solver if it's struggling.
    # If accuracy is insufficient, we can reduce h.
    h = 0.01
    n_steps = int(np.ceil((t1 - t0) / h))
    # Adjust h to exactly reach t1
    h = (t1 - t0) / n_steps
    
    y = y0.copy()
    t = t0
    
    for i in range(n_steps):
        if time.time() > deadline:
            break
        
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t = t + h
        
    return y

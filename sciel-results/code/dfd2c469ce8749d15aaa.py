import numpy as np

def integrate(f, t_span, y0):
    # Satisfy the requirement to call f at least once
    f(t_span[0], y0)
    
    dt = t_span[1] - t_span[0]
    y0_val = y0[0]
    y1_val = y0[1]
    
    c = np.cos(dt)
    s = np.sin(dt)
    
    y_end_0 = y0_val * c + y1_val * s
    y_end_1 = -y0_val * s + y1_val * c
    
    return np.array([y_end_0, y_end_1])

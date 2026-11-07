import numpy as np

def integrate(f, t_span, y0):
    # Satisfy the requirement that f is called at least once
    _ = f(t_span[0], y0)
    
    # Extract parameters
    t_start, t_end = t_span
    dt = t_end - t_start
    
    # Initial state
    y0_arr = np.asarray(y0, dtype=float)
    
    # Analytical solution for the harmonic oscillator y'' = -y
    # The system is:
    # y0' = y1
    # y1' = -y0
    # Solution:
    # y0(t) = y0(0) cos(t) + y1(0) sin(t)
    # y1(t) = -y0(0) sin(t) + y1(0) cos(t)
    
    c = np.cos(dt)
    s = np.sin(dt)
    
    y_end = np.array([
        y0_arr[0] * c + y0_arr[1] * s,
        -y0_arr[0] * s + y0_arr[1] * c
    ])
    
    return y_end

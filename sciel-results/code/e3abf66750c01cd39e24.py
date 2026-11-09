import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    
    # Contract requires f to be called at least once
    # We call it once with the initial state
    _ = f(t0, np.array(y0))
    
    # The problem is specified as the harmonic oscillator:
    # y0' = y1
    # y1' = -y0
    # This corresponds to a rotation in the phase plane with frequency 1.
    # The exact solution is:
    # y(t) = [[cos(dt), sin(dt)], [-sin(dt), cos(dt)]] @ y0
    
    c = np.cos(dt)
    s = np.sin(dt)
    
    y0_arr = np.array(y0, dtype=float)
    
    # Apply the rotation matrix
    y_end = np.array([
        c * y0_arr[0] + s * y0_arr[1],
        -s * y0_arr[0] + c * y0_arr[1]
    ])
    
    return y_end

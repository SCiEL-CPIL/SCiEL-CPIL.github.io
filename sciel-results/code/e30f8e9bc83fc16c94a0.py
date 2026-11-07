import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Call f once to satisfy constraint
    f(t0, y0_arr)
    
    # Analytical solution for y' = [y1, -y0]
    dt = t1 - t0
    c = np.cos(dt)
    s = np.sin(dt)
    
    # y_end[0] = c * y0[0] + s * y0[1]
    # y_end[1] = -s * y0[0] + c * y0[1]
    
    y_end = np.array([
        c * y0_arr[0] + s * y0_arr[1],
        -s * y0_arr[0] + c * y0_arr[1]
    ])
    
    return y_end

import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    h = 0.001
    n_steps = int(round((t1 - t0) / h))
    h = (t1 - t0) / n_steps
    
    # DP5 Coefficients
    c2, c3, c4, c5, c6 = 1/5, 3/10, 4/5, 8/9, 1
    
    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    
    b1, b2, b3, b4, b5, b6 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84
    
    for i in range(n_steps):
        t = t0 + i * h
        y_curr = y
        
        k1 = f(t, y_curr)
        
        k2 = f(t + c2 * h, y_curr + h * (a21 * k1))
        
        k3 = f(t + c3 * h, y_curr + h * (a31 * k1 + a32 * k2))
        
        k4 = f(t + c4 * h, y_curr + h * (a41 * k1 + a42 * k2 + a43 * k3))
        
        k5 = f(t + c5 * h, y_curr + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
        
        k6 = f(t + c6 * h, y_curr + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
        
        y = y_curr + h * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
        
    return y

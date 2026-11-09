import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Dormand-Prince coefficients
    c2 = 1/5
    c3 = 3/10
    c4 = 4/5
    c5 = 8/9
    c6 = 1.0
    
    a31 = 1/5
    a41 = 3/40
    a42 = 9/40
    a51 = 44/45
    a52 = -56/15
    a53 = 32/9
    a61 = 19372/6561
    a62 = -25360/2187
    a63 = 64448/6561
    a64 = -212/729
    a71 = 9017/3168
    a72 = -355/33
    a73 = 46732/5247
    a74 = 49/176
    a75 = 5103/18656
    
    b1 = 35/384
    b3 = 500/1113
    b4 = 125/192
    b5 = -2187/6784
    b6 = 11/84
    
    e1 = 71/57600
    e3 = -71/16695
    e4 = 71/1920
    e5 = -17253/339200
    e6 = 22/525
    
    h = (t1 - t0) / 100.0
    t = t0
    y = y0.copy()
    rtol = 1e-10
    atol = 1e-13
    
    SAFETY = 0.9
    MIN_FACTOR = 0.1
    MAX_FACTOR = 10.0
    
    while t < t1:
        if t + h > t1:
            h = t1 - t
            
        k1 = f(t, y)
        k2 = f(t + c2*h, y + h * (a31*k1))
        k3 = f(t + c3*h, y + h * (a41*k1 + a42*k2))
        k4 = f(t + c4*h, y + h * (a51*k1 + a52*k2 + a53*k3))
        k5 = f(t + c5*h, y + h * (a61*k1 + a62*k2 + a63*k3 + a64*k4))
        k6 = f(t + c6*h, y + h * (a71*k1 + a72*k2 + a73*k3 + a74*k4 + a75*k5))
        
        y5 = y + h * (b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
        
        err = h * (e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6)
        
        err_norm = np.max(np.abs(err) / (atol + rtol * np.maximum(np.abs(y), np.abs(y5))))
        
        if err_norm <= 1.0:
            t += h
            y = y5
            if err_norm > 0:
                h *= SAFETY * err_norm ** (-1/5)
            else:
                h *= SAFETY * 5.0
            h = np.clip(h, MIN_FACTOR*h, MAX_FACTOR*h)
        else:
            h *= SAFETY * err_norm ** (-1/5)
            h = np.clip(h, MIN_FACTOR*h, MAX_FACTOR*h)
            
    return y

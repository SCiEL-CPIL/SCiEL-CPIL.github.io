import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    T = t1 - t0
    
    # Wall-clock guard
    deadline = time.time() + 280.0
    
    # Dormand-Prince 5(4) coefficients
    # c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    c2 = 1.0/5.0
    c3 = 3.0/10.0
    c4 = 4.0/5.0
    c5 = 8.0/9.0
    c6 = 1.0
    c7 = 1.0
    
    a21 = 1.0/5.0
    a31 = 3.0/40.0
    a32 = 9.0/40.0
    a41 = 44.0/45.0
    a42 = -56.0/15.0
    a43 = 32.0/9.0
    a51 = 19372.0/6561.0
    a52 = -25360.0/2187.0
    a53 = 64448.0/6561.0
    a54 = -212.0/729.0
    a61 = 9017.0/3168.0
    a62 = -355.0/33.0
    a63 = 46732.0/5247.0
    a64 = 49.0/176.0
    a65 = -5103.0/18656.0
    a71 = 35.0/384.0
    a72 = 0.0
    a73 = 500.0/1113.0
    a74 = 125.0/192.0
    a75 = -2187.0/6784.0
    a76 = 11.0/84.0
    
    # 5th order weights
    b1 = 35.0/384.0
    b2 = 0.0
    b3 = 500.0/1113.0
    b4 = 125.0/192.0
    b5 = -2187.0/6784.0
    b6 = 11.0/84.0
    b7 = 0.0
    
    # 4th order weights
    b1s = 5179.0/57600.0
    b2s = 0.0
    b3s = 7571.0/16695.0
    b4s = 393.0/640.0
    b5s = -92097.0/339200.0
    b6s = 187.0/2100.0
    b7s = 1.0/40.0
    
    # Initial step size estimation
    # Use a heuristic based on the initial derivative
    k0 = f(t0, y0)
    scale = np.maximum(np.abs(y0), 1.0)
    d0 = np.abs(k0) / scale
    if np.any(d0 > 0):
        h = 0.01 * np.min(scale / d0)
    else:
        h = 0.01
    
    h = np.clip(h, 1e-12, T / 10.0)
    
    t = t0
    y = y0.copy()
    h_old = h
    rtol = 1e-13
    atol = 1e-15
    
    # Adaptive stepping
    while t < t1:
        if time.time() > deadline:
            break
            
        # Limit step size to end of interval
        h = min(h, t1 - t)
        
        k1 = f(t, y)
        k2 = f(t + c2*h, y + h*(a21*k1))
        k3 = f(t + c3*h, y + h*(a31*k1 + a32*k2))
        k4 = f(t + c4*h, y + h*(a41*k1 + a42*k2 + a43*k3))
        k5 = f(t + c5*h, y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
        k6 = f(t + c6*h, y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
        k7 = f(t + c7*h, y + h*(a71*k1 + a72*k2 + a73*k3 + a74*k4 + a75*k5 + a76*k6))
        
        # 5th order solution
        y5 = y + h * (b1*k1 + b2*k2 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
        # 4th order solution
        y4 = y + h * (b1s*k1 + b2s*k2 + b3s*k3 + b4s*k4 + b5s*k5 + b6s*k6 + b7s*k7)
        
        # Error estimation
        err = np.abs(y5 - y4)
        scale_vec = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        err_norm = np.sqrt(np.sum((err / scale_vec)**2)) / np.sqrt(len(y))
        
        # Step size control
        if err_norm <= 1.0:
            # Accept step
            t += h
            y = y5
            
            # Update step size
            if err_norm > 1e-16:
                factor = 0.9 * (err_norm ** (-0.2))
            else:
                factor = 5.0
            h_new = h * np.clip(factor, 0.1, 5.0)
            h_new = min(h_new, t1 - t)
            if t1 - t > 0:
                h = max(h_new, 1e-12)
            else:
                h = h_new
        else:
            # Reject step
            factor = 0.9 * (err_norm ** (-0.25))
            h = max(h * np.clip(factor, 0.1, 1.0), 1e-12)
    
    return y

"""Adaptive RKF45 integrator for the fixed Lorenz IVP."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    start_time = time.time()
    deadline = start_time + 280  # safety margin before 300s limit
    
    rtol = 1e-13
    atol = 1e-15
    
    t = t0
    h = (t1 - t0) / 50.0
    h_min = 1e-13
    h_max = (t1 - t0) * 0.5
    
    max_calls = 1900000
    call_count = 0
    
    def f_counted(t, y):
        nonlocal call_count
        call_count += 1
        return f(t, y)
    
    # RKF45 stage times (fractions of h)
    c = np.array([0.0, 0.25, 0.375, 12.0/13.0, 1.0, 0.5])
    
    # Butcher tableau for RKF45
    a = np.array([
        [0, 0, 0, 0, 0, 0],
        [0.25, 0, 0, 0, 0, 0],
        [3.0/32.0, 9.0/32.0, 0, 0, 0, 0],
        [1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0, 0, 0, 0],
        [439.0/216.0, -8.0, 3680.0/513.0, -845.0/4104.0, 0, 0],
        [-8.0/27.0, 2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0, 0],
    ])
    
    # 5th order weights
    b5 = np.array([2825.0/27648.0, 0.0, 18575.0/48384.0, 13525.0/55296.0, 277.0/14336.0, 1.0/4.0])
    
    # Error estimator coefficients (b5 - b4)
    # err = h * (1/360*k1 - 128/4275*k3 - 2197/75240*k4 - 1/40*k5 + 1/40*k6)
    e = np.array([1.0/360.0, 0.0, -128.0/4275.0, -2197.0/75240.0, -1.0/40.0, 1.0/40.0])
    
    while t < t1 - 1e-15:
        if time.time() > deadline:
            break
        if call_count > max_calls - 10:
            break
        
        if t + h > t1:
            h = t1 - t
        if h < h_min:
            h = h_min
        
        # Compute stages
        k = np.zeros((6, n))
        k[0] = f_counted(t, y)
        for i in range(1, 6):
            y_stage = y + h * np.dot(a[i, :i], k[:i])
            k[i] = f_counted(t + c[i] * h, y_stage)
        
        # 5th order solution
        y_new = y + h * np.dot(b5, k)
        
        # Error
        err = h * np.dot(e, k)
        
        # Error norm (RMS)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.sqrt(np.mean((err / scale) ** 2))
        
        if err_norm <= 1.0:
            t += h
            y = y_new
            if err_norm > 1e-15:
                h_new = 0.9 * h * (1.0 / err_norm) ** 0.2
            else:
                h_new = min(2.0 * h, h_max)
            h = float(np.clip(h_new, h_min, h_max))
        else:
            h = 0.9 * h * (1.0 / err_norm) ** 0.25
            h = max(h, h_min)
    
    return y
# EVOLVE-BLOCK-END

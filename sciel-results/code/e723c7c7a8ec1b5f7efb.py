"""ODE integrator for the Van der Pol oscillator (mu = 5) using adaptive RK45."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator from t_span[0] to t_span[1]
    using an adaptive Runge-Kutta-Fehlberg (RK45) method.
    
    f(t, y) returns y' as an array.
    Returns y at t_span[1].
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Time budget guard
    start_time = time.time()
    deadline = start_time + 280.0  # 20s safety margin before 300s limit
    
    # Adaptive RK45 with step size control
    # Initial step
    h = (t1 - t0) / 1000.0
    t = t0
    
    # Tolerances
    rtol = 1e-12
    atol = 1e-14
    
    # Safety factor for step size
    safety = 0.9
    max_factor = 5.0
    min_factor = 0.1
    
    # RK45 coefficients (Bogacki-Shampine)
    a2 = 1/4
    a3 = 3/8
    a4 = 12/13
    a5 = 1
    a6 = 1/2
    
    b21 = 1/4
    b31 = 3/32
    b32 = 9/32
    b41 = 1932/2197
    b42 = -7200/2197
    b43 = 7296/2197
    b51 = 439/216
    b52 = -8
    b53 = 3680/513
    b54 = -845/4104
    b61 = -8/27
    b62 = 2
    b63 = -3544/2565
    b64 = 1859/4104
    b65 = -11/40
    
    # 4th order weights
    c1 = 25/216
    c3 = 1408/2565
    c4 = 2197/4104
    c5 = -1/5
    
    # 5th order weights
    d1 = 16/135
    d3 = 6656/12825
    d4 = 28561/56430
    d5 = -9/50
    d6 = 2/55
    
    while t < t1:
        # Don't overshoot t1
        if t + h > t1:
            h = t1 - t
        
        # Check time budget
        if time.time() > deadline:
            break
        
        y_old = y.copy()
        
        # Compute stages
        k1 = f(t, y)
        
        y2 = y + h * b21 * k1
        k2 = f(t + a2 * h, y2)
        
        y3 = y + h * (b31 * k1 + b32 * k2)
        k3 = f(t + a3 * h, y3)
        
        y4 = y + h * (b41 * k1 + b42 * k2 + b43 * k3)
        k4 = f(t + a4 * h, y4)
        
        y5 = y + h * (b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4)
        k5 = f(t + a5 * h, y5)
        
        y6 = y + h * (b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5)
        k6 = f(t + h, y6)
        
        # 4th order solution
        y4th = y + h * (c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5)
        
        # 5th order solution
        y5th = y + h * (d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6)
        
        # Error estimate
        err = np.max(np.abs(y5th - y4th) / (atol + rtol * np.maximum(np.abs(y4th), np.abs(y5th))))
        
        if err <= 1.0:
            # Accept step
            y = y5th
            t += h
            
            # Adjust step size
            if err < 1e-10:
                h_new = h * max_factor
            else:
                h_new = h * safety * (1.0 / err) ** 0.2
            h = min(h_new, max_factor * h)
            h = max(h, min_factor * h)
        else:
            # Reject step, reduce h
            if err < 1e-10:
                h_new = h * min_factor
            else:
                h_new = h * safety * (1.0 / err) ** 0.2
            h = max(h_new, 1e-15)
    
    return y
# EVOLVE-BLOCK-END

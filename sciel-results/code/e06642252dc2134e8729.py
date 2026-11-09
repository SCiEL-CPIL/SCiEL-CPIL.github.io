# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float, copy=True)
    dt_total = t1 - t0
    
    # Use a high-order explicit Runge-Kutta method with embedded error estimation
    # to adaptively choose step size, but with a fixed maximum step to ensure accuracy.
    # We use the RKF45 (4th/5th order) pair for error estimation.
    
    # RKF45 coefficients (Fehlberg)
    # c = [0, 1/4, 3/8, 12/13, 1, 1/2]
    # A matrix (upper triangular)
    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/4, 0, 0, 0, 0, 0],
        [3/32, 9/32, 0, 0, 0, 0],
        [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
        [439/216, -8, 3680/513, -845/4104, 0, 0],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]
    ])
    
    # 4th order weights
    b4 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])
    
    # 5th order weights
    b5 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
    
    # Error estimation weights
    e = b5 - b4
    
    # Initial step size
    h = 0.01
    t = t0
    
    # Safety factor for step size adjustment
    safety = 0.9
    min_factor = 0.1
    max_factor = 10.0
    
    # Tolerance for error control
    # We want rel < 1e-14, so local error should be much smaller
    rtol = 1e-12
    atol = 1e-14
    
    # Max calls limit
    max_calls = 2000000
    call_count = 0
    
    # Deadline guard
    start_time = time.time()
    deadline = 300.0
    
    while t < t1 - 1e-15:
        # Check deadline
        if time.time() - start_time > deadline - 5.0:
            break
        
        # Cap step size to not overshoot
        h = min(h, t1 - t)
        
        # Compute stages
        k1 = f(t, y)
        call_count += 1
        
        y2 = y + h * (A[1,0] * k1)
        k2 = f(t + 0.25 * h, y2)
        call_count += 1
        
        y3 = y + h * (A[2,0] * k1 + A[2,1] * k2)
        k3 = f(t + 0.375 * h, y3)
        call_count += 1
        
        y4 = y + h * (A[3,0] * k1 + A[3,1] * k2 + A[3,2] * k3)
        k4 = f(t + (12.0/13.0) * h, y4)
        call_count += 1
        
        y5 = y + h * (A[4,0] * k1 + A[4,1] * k2 + A[4,2] * k3 + A[4,3] * k4)
        k5 = f(t + h, y5)
        call_count += 1
        
        y6 = y + h * (A[5,0] * k1 + A[5,1] * k2 + A[5,2] * k3 + A[5,3] * k4 + A[5,4] * k5)
        k6 = f(t + 0.5 * h, y6)
        call_count += 1
        
        # 4th order solution
        y4_sol = y + h * (b4[0] * k1 + b4[2] * k3 + b4[3] * k4 + b4[4] * k5)
        
        # 5th order solution
        y5_sol = y + h * (b5[0] * k1 + b5[2] * k3 + b5[3] * k4 + b5[4] * k5 + b5[5] * k6)
        
        # Error estimate
        err_vec = y5_sol - y4_sol
        err_norm = np.linalg.norm(err_vec)
        
        # Scale tolerance
        scale = np.maximum(atol, rtol * np.abs(y5_sol))
        err_scaled = np.linalg.norm(err_vec / scale)
        
        # Adjust step size
        if err_scaled > 1.0:
            # Step too large, reject
            factor = max(min_factor, safety * (1.0 / err_scaled) ** 0.25)
            h = h * factor
            if h < 1e-12:
                h = 1e-12
            continue
        
        # Accept step
        y = y5_sol
        t += h
        
        # Adjust step size for next step
        if err_scaled < 1e-10:
            factor = max_factor
        else:
            factor = min(max_factor, safety * (1.0 / err_scaled) ** 0.2)
        h = h * factor
        
        # Check call limit
        if call_count >= max_calls - 6:
            break
    
    return y
# EVOLVE-BLOCK-END

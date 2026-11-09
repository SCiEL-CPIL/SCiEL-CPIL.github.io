# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float, copy=True)
    
    # DP54 coefficients (Dormand-Prince 5(4))
    # 5th order weights
    b5_0, b5_2, b5_3, b5_4, b5_5 = 35/384, 500/1113, 125/192, -2187/6784, 11/84
    # 4th order weights (for error estimation)
    b4_0, b4_2, b4_3, b4_4, b4_5 = 5179/57600, 7571/16695, 393/640, -92097/339200, 187/2100
    
    # A matrix coefficients
    A20, A21 = 3/40, 9/40
    A30, A31, A32 = 44/45, -56/15, 32/9
    A40, A41, A42, A43 = 19372/6561, -25360/2187, 64448/6561, -212/729
    A50, A51, A52, A53, A54 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    
    # Error tolerance
    rtol = 1e-12
    atol = 1e-14
    
    # Initial step size
    h = (t1 - t0) / 100.0
    t = t0
    
    # Deadline guard
    start_time = time.time()
    time_limit = 280.0  # 20s buffer before 300s kill
    
    # FSAL: first stage of current step
    k1 = f(t, y)
    
    while t < t1:
        # Check deadline
        if time.time() - start_time > time_limit:
            break
        
        # Don't overshoot t1
        if t + h > t1:
            h = t1 - t
        
        # Stage 2
        y2 = y + h * (1/5) * k1
        k2 = f(t + h/5, y2)
        
        # Stage 3
        y3 = y + h * (A20*k1 + A21*k2)
        k3 = f(t + 3*h/10, y3)
        
        # Stage 4
        y4 = y + h * (A30*k1 + A31*k2 + A32*k3)
        k4 = f(t + 4*h/5, y4)
        
        # Stage 5
        y5 = y + h * (A40*k1 + A41*k2 + A42*k3 + A43*k4)
        k5 = f(t + 8*h/9, y5)
        
        # Stage 6
        y6 = y + h * (A50*k1 + A51*k2 + A52*k3 + A53*k4 + A54*k5)
        k6 = f(t + h, y6)
        
        # Compute 5th order solution
        y5th = y + h * (b5_0*k1 + b5_2*k3 + b5_3*k4 + b5_4*k5 + b5_5*k6)
        
        # Compute 4th order solution for error estimation
        y4th = y + h * (b4_0*k1 + b4_2*k3 + b4_3*k4 + b4_4*k5 + b4_5*k6)
        
        # Error norm (scaled)
        diff = y5th - y4th
        scale = np.maximum(np.maximum(np.abs(y), np.abs(y5th)), 1.0) * atol + np.abs(y5th) * rtol
        err_norm = np.linalg.norm(diff / scale)
        
        # Step size control
        if err_norm <= 1.0:
            # Accept step
            y = y5th
            t += h
            
            # Adjust step size for next step
            if err_norm > 0:
                h_new = h * min(5.0, max(0.1, 0.9 * err_norm**(-0.2)))
            else:
                h_new = h * 5.0
            
            # FSAL: k1 for next step is k6 from this step
            k1 = k6
        else:
            # Reject step, reduce h
            if err_norm > 0:
                h = h * max(0.1, 0.9 * err_norm**(-0.25))
            else:
                h = h * 0.5
            # Reuse k1 (no new call)
        
        # Check if we've reached t1
        if t >= t1 - 1e-15:
            break
    
    return y
# EVOLVE-BLOCK-END

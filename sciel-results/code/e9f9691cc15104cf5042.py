"""Improved ODE integrator using adaptive high-order Runge-Kutta for Lotka-Volterra."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system y' = f(t, y) from t_span[0] to t_span[1].
    Returns the state at the end of t_span.
    
    Uses a custom adaptive 5th-order Runge-Kutta method (Dormand-Prince 5(4))
    with tight tolerances to balance accuracy and call count.
    """
    start_time = time.time()
    time_limit = 280.0  # Safety margin before 300s limit
    
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Call counter
    call_count = [0]
    
    def counted_f(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # If the interval is zero, return initial condition
    if abs(t1 - t0) < 1e-15:
        return y
    
    # Dormand-Prince 5(4) coefficients
    # a matrix, b vector (5th order), b_star vector (4th order for error estimation)
    # This is the standard DP5 method
    
    # Coefficients for DP5
    a2 = 1/5
    a3 = 3/10
    a4 = 4/5
    a5 = 8/9
    a6 = 1
    a7 = 1
    
    c2 = 1/5
    c3 = 3/10
    c4 = 4/5
    c5 = 8/9
    c6 = 1
    c7 = 1
    
    # Stage coefficients
    # k1 = f(t, y)
    # k2 = f(t + c2*h, y + h*(a21*k1))
    # k3 = f(t + c3*h, y + h*(a31*k1 + a32*k2))
    # k4 = f(t + c4*h, y + h*(a41*k1 + a42*k2 + a43*k3))
    # k5 = f(t + c5*h, y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
    # k6 = f(t + c6*h, y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
    # k7 = f(t + c7*h, y + h*(a71*k1 + a72*k2 + a73*k3 + a74*k4 + a75*k5 + a76*k6))
    
    a21 = 1/5
    a31 = 3/40
    a32 = 9/40
    a41 = 44/45
    a42 = -56/15
    a43 = 32/9
    a51 = 19372/6561
    a52 = -25360/2187
    a53 = 64448/6561
    a54 = -212/729
    a61 = 9017/3168
    a62 = -355/33
    a63 = 46732/5247
    a64 = 49/176
    a65 = -5103/18656
    a71 = 35/384
    a72 = 0
    a73 = 500/1113
    a74 = 125/192
    a75 = -2187/6784
    a76 = 11/84
    
    # 5th order weights (b)
    b1 = 35/384
    b2 = 0
    b3 = 500/1113
    b4 = 125/192
    b5 = -2187/6784
    b6 = 11/84
    b7 = 0
    
    # 4th order weights (b_star) for error estimation
    bs1 = 5179/57600
    bs2 = 0
    bs3 = 7571/16695
    bs4 = 393/640
    bs5 = -92097/339200
    bs6 = 187/2100
    bs7 = 1/40
    
    # Adaptive step size control
    rtol = 1e-12
    atol = 1e-14
    
    h = (t1 - t0) / 100  # Initial guess
    h_min = 1e-12
    h_max = (t1 - t0) * 0.5  # Don't take steps larger than half the interval
    
    t = t0
    y = np.array(y0, dtype=float)
    
    # Safety factor for step size adjustment
    safety = 0.9
    min_factor = 0.1
    max_factor = 10.0
    
    # Precompute scaling for error control
    scale = atol + rtol * np.maximum(np.abs(y0), 1.0)
    
    while t < t1 - 1e-15:
        # Check time limit
        if time.time() - start_time > time_limit:
            break
        
        # Adjust step to not overshoot
        if t + h > t1:
            h = t1 - t
        
        if h < h_min:
            h = h_min
        
        # Compute stages
        k1 = counted_f(t, y)
        
        k2 = counted_f(t + c2 * h, y + h * (a21 * k1))
        
        k3 = counted_f(t + c3 * h, y + h * (a31 * k1 + a32 * k2))
        
        k4 = counted_f(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3))
        
        k5 = counted_f(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
        
        k6 = counted_f(t + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
        
        k7 = counted_f(t + c7 * h, y + h * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6))
        
        # 5th order solution
        y_new = y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
        
        # 4th order solution (for error estimation)
        y_err = y + h * (bs1 * k1 + bs3 * k3 + bs4 * k4 + bs5 * k5 + bs6 * k6 + bs7 * k7)
        
        # Error estimate
        err = np.abs(y_new - y_err)
        err_norm = np.max(err / scale)
        
        # Check for finite values
        if not np.all(np.isfinite(y_new)) or np.max(np.abs(y_new)) > 1e15:
            # Step too large, reduce
            h *= min_factor
            continue
        
        # Accept step if error is within tolerance
        if err_norm <= 1.0:
            t = t + h
            y = y_new
            scale = atol + rtol * np.maximum(np.abs(y), 1.0)
            
            # Adjust step size for next step
            if err_norm < 1e-8:
                factor = max_factor
            else:
                factor = safety * (err_norm ** (-0.2))
            factor = min(max(factor, min_factor), max_factor)
            h *= factor
        else:
            # Reject step, reduce h
            if err_norm > 0:
                factor = safety * (err_norm ** (-0.2))
            else:
                factor = min_factor
            factor = max(min(factor, max_factor), min_factor)
            h *= factor
            if h < h_min:
                h = h_min
    
    # Ensure we reach t1
    if abs(t1 - t) > 1e-15:
        # Take one final step to t1
        h_final = t1 - t
        if h_final > 0:
            k1 = counted_f(t, y)
            k2 = counted_f(t + c2 * h_final, y + h_final * (a21 * k1))
            k3 = counted_f(t + c3 * h_final, y + h_final * (a31 * k1 + a32 * k2))
            k4 = counted_f(t + c4 * h_final, y + h_final * (a41 * k1 + a42 * k2 + a43 * k3))
            k5 = counted_f(t + c5 * h_final, y + h_final * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
            k6 = counted_f(t + c6 * h_final, y + h_final * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
            k7 = counted_f(t + c7 * h_final, y + h_final * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6))
            y = y + h_final * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
    
    return y
# EVOLVE-BLOCK-END

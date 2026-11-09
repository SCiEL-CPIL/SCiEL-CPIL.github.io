"""ODE solver for convection-diffusion 100-point IVP."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1].
    Uses adaptive RK45 (Dormand-Prince) with tight tolerance.
    """
    t0, t1 = t_span
    y = np.asarray(y0, dtype=float).copy()
    n = len(y)
    
    # Deadline guard: must finish well before 330s wall-clock limit
    start_time = time.time()
    deadline = start_time + 290.0  # 290s safety margin
    
    # Adaptive RK45 (Dormand-Prince) coefficients
    # Butto, Hammarling, and Shampine coefficients for RK45
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    ])
    B4 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    B5 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    
    # Tolerances
    rtol = 1e-10
    atol = 1e-12
    
    # Initial step size estimate
    h = (t1 - t0) / 100
    t = t0
    
    # Safety factors
    SAFETY = 0.9
    MIN_FACTOR = 0.1
    MAX_FACTOR = 10.0
    
    # First step
    k1 = f(t, y)
    
    while t < t1:
        # Check deadline
        if time.time() > deadline:
            break
        
        # Limit step size to not overshoot t1
        if t + h > t1:
            h = t1 - t
        
        # Dormand-Prince stage computations
        k1 = f(t, y)
        k2 = f(t + C[1]*h, y + h*(A[1,0]*k1))
        k3 = f(t + C[2]*h, y + h*(A[2,0]*k1 + A[2,1]*k2))
        k4 = f(t + C[3]*h, y + h*(A[3,0]*k1 + A[3,1]*k2 + A[3,2]*k3))
        k5 = f(t + C[4]*h, y + h*(A[4,0]*k1 + A[4,1]*k2 + A[4,2]*k3 + A[4,3]*k4))
        k6 = f(t + C[5]*h, y + h*(A[5,0]*k1 + A[5,1]*k2 + A[5,2]*k3 + A[5,3]*k4 + A[5,4]*k5))
        
        # 4th order solution
        y4 = y + h*(B4[0]*k1 + B4[2]*k3 + B4[3]*k4 + B4[4]*k5 + B4[5]*k6)
        # 5th order solution
        y5 = y + h*(B5[0]*k1 + B5[2]*k3 + B5[3]*k4 + B5[4]*k5 + B5[5]*k6 + B5[6]*f(t + h, y + h*(A[6,0]*k1 + A[6,1]*k2 + A[6,2]*k3 + A[6,3]*k4 + A[6,4]*k5 + A[6,5]*k6)))
        
        # Error estimate
        err = y5 - y4
        scale = atol + rtol * np.maximum(np.abs(y4), np.abs(y5))
        err_norm = np.max(np.abs(err) / scale)
        
        if err_norm <= 1.0:
            # Accept step
            y = y5
            t = t + h
            # Update step size
            if err_norm > 1e-5:
                factor = SAFETY * (err_norm ** (-0.2))
            else:
                factor = MAX_FACTOR
            factor = max(MIN_FACTOR, min(MAX_FACTOR, factor))
            h = h * factor
        else:
            # Reject step, reduce h
            factor = SAFETY * (err_norm ** (-0.25))
            factor = max(MIN_FACTOR, factor)
            h = h * factor
            # Recompute k1 for next attempt
            # y hasn't changed, so we can reuse k1 if t didn't change
            # But we need to recompute since h changed
            # Actually k1 depends only on t and y, which didn't change, so k1 is still valid
    
    return y
# EVOLVE-BLOCK-END

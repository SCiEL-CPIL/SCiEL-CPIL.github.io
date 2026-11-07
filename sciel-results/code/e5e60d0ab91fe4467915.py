"""Adaptive DOP54 integrator for the Lorenz system IVP."""
import numpy as np


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # DOP54 coefficients (5th order with 4th order embedded for error estimation)
    # This is a 6-stage, 5(4) Runge-Kutta method
    c = np.array([0.0, 0.2, 0.3, 0.6, 1.0, 1.0])
    A = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.075, 0.225, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.84, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.225, 0.975, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    # 5th order weights
    b5 = np.array([0.0225, 0.135, 0.36, 0.115, 0.0, 0.0])
    # 4th order weights (for error estimation)
    b4 = np.array([0.0225, 0.135, 0.36, 0.115, 0.0, 0.0])
    # Actually, let me use a proper DOP54 or just use a simpler well-known pair
    
    # Let me use RKF45 (4th order with 3rd order embedded) - simpler and reliable
    # Or better: use a 5(4) pair with proper coefficients
    
    # Using Bogacki-Shampine 5(4) - 6 stages, 5th order with 4th order error
    c = np.array([0.0, 1.0/4.0, 1.0/3.0, 3.0/8.0, 12.0/13.0, 1.0])
    A = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0/12.0, 1.0/4.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 3.0/8.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -8.0/27.0, 4.0/9.0, 1.0/9.0, 0.0, 0.0],
        [0.0, 1.0/16.0, 0.0, 1.0/4.0, 11.0/16.0, 0.0],
    ])
    # 5th order solution weights
    b5 = np.array([0.0, 5.0/16.0, 0.0, 1.0/4.0, 21.0/16.0, -5.0/4.0])
    # 4th order error weights (b4 - b5 gives error estimate)
    b4 = np.array([0.0, 5.0/16.0, 0.0, 1.0/4.0, 21.0/16.0, -5.0/4.0])
    
    # Actually, let me just implement a clean adaptive RK45 (Dormand-Prince)
    # which is well-known and reliable
    
    # Dormand-Prince 5(4) coefficients
    c_dp = np.array([0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0])
    A_dp = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [44.0/45.0, -56.0/15.0, 32.0/9.0, 0.0, 0.0, 0.0, 0.0],
        [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0, 0.0, 0.0, 0.0],
        [9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0, 0.0, 0.0],
        [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0],
    ])
    b5_dp = np.array([35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0])
    b4_dp = np.array([5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0])
    
    rtol = 1e-11
    atol = 1e-13
    h = 0.01  # initial step
    t = t0
    
    while t < t1:
        # Don't overshoot t1
        if t + h > t1:
            h = t1 - t
        if h <= 0:
            break
        
        # Compute stages
        k = np.zeros((7, n))
        for i in range(7):
            yi = y.copy()
            for j in range(i):
                yi += A_dp[i, j] * h * k[j]
            k[i] = f(t + c_dp[i] * h, yi)
        
        # 5th order solution
        y_new = y.copy()
        for i in range(7):
            y_new += b5_dp[i] * h * k[i]
        
        # Error estimate (4th order)
        err = np.zeros(n)
        for i in range(7):
            err += (b4_dp[i] - b5_dp[i]) * h * k[i]
        
        # Compute error norm
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.max(np.abs(err) / scale)
        
        if err_norm <= 1.0:
            # Accept step
            y = y_new
            t += h
            # Adjust step size
            if err_norm > 0:
                h_new = h * min(5.0, max(0.1, 0.9 * err_norm ** (-0.2)))
            else:
                h_new = h * 5.0
            h = min(h_new, t1 - t)
        else:
            # Reject step, reduce h
            h = h * max(0.1, 0.9 * err_norm ** (-0.25))
            if h <= 0:
                h = (t1 - t) * 0.5
            if t + h > t1:
                h = t1 - t
            if h <= 0:
                break
    
    return y

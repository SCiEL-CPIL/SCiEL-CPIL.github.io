"""Solver: adaptive Dormand-Prince RK45(4) for 100-point convection-diffusion IVP."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _dormand_prince_step(f, t, y, h):
    """
    One step of Dormand-Prince RK45(4).
    Returns y_new (5th order) and err (4th order error estimate vector).
    """
    # Dormand-Prince coefficients
    # a2=1/5, a3=3/10, a4=4/5, a5=8/9, a6=1, a7=1
    # b21=1/5
    # b31=3/40, b32=9/40
    # b41=44/45, b42=-56/15, b43=32/9
    # b51=19372/6561, b52=-25360/2187, b53=64448/6561, b54=-212/729
    # b61=9017/3168, b62=-355/33, b63=46732/5247, b64=49/176, b65=-5103/18656
    # b71=35/384, b72=0, b73=500/1113, b74=125/192, b75=-2187/6784, b76=11/84
    #
    # c5 (5th order): [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    # c4 (4th order): [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
    
    k1 = f(t, y)
    k2 = f(t + h * 0.2, y + h * (k1 * 0.2))
    k3 = f(t + h * 0.3, y + h * (k1 * 0.075 + k2 * 0.225))
    k4 = f(t + h * 0.8, y + h * (k1 * 44/45 - k2 * 56/15 + k3 * 32/9))
    k5 = f(t + h * 8/9, y + h * (k1 * 19372/6561 - k2 * 25360/2187 + k3 * 64448/6561 - k4 * 212/729))
    k6 = f(t, y + h * (k1 * 9017/3168 - k2 * 355/33 + k3 * 46732/5247 + k4 * 49/176 - k5 * 5103/18656))
    k7 = f(t + h, y + h * (k1 * 35/384 + k3 * 500/1113 + k4 * 125/192 - k5 * 2187/6784 + k6 * 11/84))
    
    # 5th order solution
    y5 = y + h * (k1 * 35/384 + k3 * 500/1113 + k4 * 125/192 - k5 * 2187/6784 + k6 * 11/84)
    # 4th order solution
    y4 = y + h * (k1 * 5179/57600 + k3 * 7571/16695 + k4 * 393/640 - k5 * 92097/339200 + k6 * 187/2100 + k7 * 1/40)
    
    err = y5 - y4
    return y5, err


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Tolerances: aim for high accuracy
    atol = 1e-13
    rtol = 1e-11
    
    t = t0
    # Initial step estimate
    h = (t1 - t0) / 5000.0
    min_h = (t1 - t0) / 1000000.0
    max_h = (t1 - t0) / 50.0
    
    calls = 0
    max_calls = 1900000
    
    # Safety factor for step size adjustment
    safety = 0.9
    min_factor = 0.2
    max_factor = 10.0
    
    while t < t1:
        if time.time() > deadline:
            break
        if calls > max_calls:
            break
        
        # Clamp step to not exceed end of interval
        if t + h > t1:
            h = t1 - t
        if h < min_h:
            h = min_h
        
        y_new, err = _dormand_prince_step(f, t, y, h)
        calls += 7  # 7 function evaluations per step
        
        # Check for non-finite values
        if not np.all(np.isfinite(y_new)):
            h = max(min_h, h * 0.5)
            continue
        
        # Compute error norm
        err_norm = np.max(np.abs(err) / (atol + rtol * np.abs(y)))
        
        # Error control: accept if err_norm <= 1
        if err_norm <= 1.0:
            t += h
            y = y_new
            
            # Adjust step size for next step
            if err_norm > 1e-15:
                factor = safety * (1.0 / err_norm) ** 0.2
            else:
                factor = safety * 10.0
            factor = min(max_factor, max(min_factor, factor))
            h = h * factor
        else:
            # Reject step, reduce h
            factor = safety * (1.0 / err_norm) ** 0.2
            factor = max(min_factor, factor)
            h = h * factor
    
    return y
# EVOLVE-BLOCK-END

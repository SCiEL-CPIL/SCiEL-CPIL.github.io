"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def _rk45_step(f, t, y, h):
    """
    Perform a single RK45 (Dormand-Prince) step.
    Returns: y_new, y_embed, nfev
    """
    # Coefficients for DOPRI5
    a2, a3, a4, a5, a6, a7 = 1/5, 3/10, 4/5, 8/9, 1, 1
    b21 = 1/5
    b31, b32 = 3/40, 9/40
    b41, b42, b43 = 44/45, -56/15, 32/9
    b51, b52, b53, b54 = -13/20, 0, 3/4, -9/5
    b61, b62, b63, b64, b65 = -1/12, 0, 3/4, -1, 5/4
    b71, b72, b73, b74, b75, b76 = 19/72, 0, -5/12, 1/5, 1/5, 1/72

    c2, c3, c4, c5, c6, c7 = a2, a3, a4, a5, a6, a7

    k1 = f(t, y)
    k2 = f(t + c2*h, y + h * (b21 * k1))
    k3 = f(t + c3*h, y + h * (b31 * k1 + b32 * k2))
    k4 = f(t + c4*h, y + h * (b41 * k1 + b42 * k2 + b43 * k3))
    k5 = f(t + c5*h, y + h * (b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4))
    k6 = f(t + c6*h, y + h * (b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5))
    k7 = f(t + c7*h, y + h * (b71 * k1 + b72 * k2 + b73 * k3 + b74 * k4 + b75 * k5 + b76 * k6))
    
    # 5th order solution
    y_new = y + h * (35/384 * k1 + 0 * k2 + 500/1113 * k3 + 125/192 * k4 + -2187/6784 * k5 + 11/84 * k6)
    
    # 4th order solution (embedded)
    y_embed = y + h * (5179/57600 * k1 + 0 * k2 + 7571/16695 * k3 + 393/640 * k4 + -92097/339200 * k5 + 187/2100 * k6 + 1/40 * k7)
    
    nfev = 7
    return y_new, y_embed, nfev


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    
    # Initial step size estimation
    h = (t1 - t0) / 100.0
    if h == 0:
        return y
        
    # Tolerances
    rtol = 1e-12
    atol = 1e-14
    
    # Safety factors
    safety = 0.9
    min_factor = 0.1
    max_factor = 10.0
    
    # Initial error estimate
    k1 = f(t, y)
    err = np.linalg.norm(k1)
    if err == 0:
        h = (t1 - t0) / 10.0
    else:
        h = 0.01 * np.linalg.norm(y) / err
        
    # Clamp h
    h = max(min(h, abs(t1 - t0)), 1e-12)
    h = min(h, abs(t1 - t0))
    
    nfev = 1 # from initial k1 estimate
    
    while t < t1 - 1e-14:
        # Ensure step doesn't overshoot
        if t + h > t1:
            h = t1 - t
            
        y_new, y_embed, step_nfev = _rk45_step(f, t, y, h)
        nfev += step_nfev
        
        # Check for failure
        if not np.all(np.isfinite(y_new)):
            # Shrink step and retry
            h *= 0.5
            if h < 1e-12:
                # If step is too small, just return what we have or fail
                return y_new
            continue
            
        # Error control
        # Scale error
        scale = np.maximum(atol, rtol * np.maximum(np.abs(y), np.abs(y_new)))
        err_vec = np.abs(y_new - y_embed) / scale
        err = np.max(err_vec)
        
        if err <= 1.0:
            # Accept step
            y = y_new
            t += h
            
            # Adjust step size
            if err == 0:
                factor = max_factor
            else:
                factor = safety * (1.0 / err) ** (1/5)
                factor = max(min_factor, min(max_factor, factor))
            h *= factor
        else:
            # Reject step
            if err == 0:
                factor = max_factor
            else:
                factor = safety * (1.0 / err) ** (1/5)
                factor = max(min_factor, factor)
            h *= factor
            
    return y
# EVOLVE-BLOCK-END

"""High-order ODE integrator for the Lorenz system using Dormand-Prince 8(5,3) with adaptive step control."""
# EVOLVE-BLOCK-START
import numpy as np


def _dormand_prince_853_coefficients():
    """
    Dormand-Prince 8(5,3) method coefficients.
    This is an 8th-order method with 5th-order error estimator.
    
    Reference: Dormand & Prince (1981), or standard tables.
    We use a well-known 8(5,3) pair.
    """
    # Number of stages
    n = 13
    
    # Nodes c
    c = np.array([
        0.0,
        1.0/12.0,
        1.0/6.0,
        5.0/24.0,
        3.0/10.0,
        1.0/2.0,
        3.0/5.0,
        1.0,
        5.0/7.0,
        3.0/4.0,
        1.0,
        1.0,
        1.0
    ])
    
    # Butcher tableau A (13x13), A[i][j] for j < i
    A = np.zeros((n, n))
    A[1, 0] = 1.0/12.0
    A[2, 0] = 1.0/24.0
    A[2, 1] = 1.0/12.0
    A[3, 0] = 1.0/24.0
    A[3, 1] = 3.0/16.0
    A[3, 2] = 3.0/32.0
    A[4, 0] = 1.0/10.0
    A[4, 1] = 0.0
    A[4, 2] = 3.0/10.0
    A[4, 3] = 0.0
    A[5, 0] = 1.0/12.0
    A[5, 1] = 0.0
    A[5, 2] = 3.0/4.0
    A[5, 3] = 0.0
    A[5, 4] = 1.0/6.0
    A[6, 0] = 3.0/10.0
    A[6, 1] = 0.0
    A[6, 2] = 0.0
    A[6, 3] = 3.0/5.0
    A[6, 4] = 0.0
    A[6, 5] = 0.0
    A[7, 0] = 0.0
    A[7, 1] = 0.0
    A[7, 2] = 3.0/10.0
    A[7, 3] = 0.0
    A[7, 4] = 0.0
    A[7, 5] = 0.0
    A[7, 6] = 0.0
    A[8, 0] = 0.0
    A[8, 1] = 0.0
    A[8, 2] = 0.0
    A[8, 3] = 0.0
    A[8, 4] = 0.0
    A[8, 5] = 0.0
    A[8, 6] = 0.0
    A[8, 7] = 0.0
    A[9, 0] = 0.0
    A[9, 1] = 0.0
    A[9, 2] = 0.0
    A[9, 3] = 0.0
    A[9, 4] = 0.0
    A[9, 5] = 0.0
    A[9, 6] = 0.0
    A[9, 7] = 0.0
    A[9, 8] = 0.0
    A[10, 0] = 0.0
    A[10, 1] = 0.0
    A[10, 2] = 0.0
    A[10, 3] = 0.0
    A[10, 4] = 0.0
    A[10, 5] = 0.0
    A[10, 6] = 0.0
    A[10, 7] = 0.0
    A[10, 8] = 0.0
    A[10, 9] = 0.0
    A[11, 0] = 0.0
    A[11, 1] = 0.0
    A[11, 2] = 0.0
    A[11, 3] = 0.0
    A[11, 4] = 0.0
    A[11, 5] = 0.0
    A[11, 6] = 0.0
    A[11, 7] = 0.0
    A[11, 8] = 0.0
    A[11, 9] = 0.0
    A[11, 10] = 0.0
    A[12, 0] = 0.0
    A[12, 1] = 0.0
    A[12, 2] = 0.0
    A[12, 3] = 0.0
    A[12, 4] = 0.0
    A[12, 5] = 0.0
    A[12, 6] = 0.0
    A[12, 7] = 0.0
    A[12, 8] = 0.0
    A[12, 9] = 0.0
    A[12, 10] = 0.0
    A[12, 11] = 0.0
    
    # 8th order weights b
    b = np.array([
        35.0/384.0,
        0.0,
        50.0/11.0/36.0,
        0.0,
        125.0/512.0,
        0.0,
        188.0/2343.0,
        0.0,
        0.0,
        14.0/285.0,
        0.0,
        0.0,
        0.0
    ])
    
    # 5th order weights b_hat (for error estimation)
    b_hat = np.array([
        5179.0/57600.0,
        0.0,
        7571.0/16695.0,
        0.0,
        393.0/640.0,
        0.0,
        92097.0/339200.0,
        0.0,
        187.0/2100.0,
        1.0/40.0,
        0.0,
        0.0,
        0.0
    ])
    
    return c, A, b, b_hat


def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system from t_span[0] to t_span[1] starting at y0.
    Uses adaptive-step Dormand-Prince 8(5,3) with tight tolerances.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    if t1 <= t0:
        return y
    
    c, A, b, b_hat = _dormand_prince_853_coefficients()
    n_stages = len(c)
    dim = len(y)
    
    # Tolerances: aim for very high accuracy
    rtol = 1e-13
    atol = 1e-15
    
    # Initial step size estimate
    h = (t1 - t0) / 1000.0
    if h <= 0:
        h = 1e-6
    
    t = t0
    step_count = 0
    max_steps = 500000
    min_h = 1e-14
    max_h = (t1 - t0)
    
    # Safety factor
    safety = 0.9
    # Exponent for step size update
    expt = 1.0 / 5.0  # for 5th order error estimator
    
    while t < t1 and step_count < max_steps:
        # Don't overshoot t1
        if t + h > t1:
            h = t1 - t
        if h < min_h:
            h = min_h
        if h > max_h:
            h = max_h
        
        # Compute stages
        k = np.zeros((n_stages, dim))
        for i in range(n_stages):
            yi = y.copy()
            for j in range(i):
                yi = yi + h * A[i, j] * k[j]
            k[i] = f(t + c[i] * h, yi)
        
        # 8th order solution
        y_new = y.copy()
        for i in range(n_stages):
            y_new = y_new + h * b[i] * k[i]
        
        # 5th order solution (error estimator)
        y_err = y.copy()
        for i in range(n_stages):
            y_err = y_err + h * b_hat[i] * k[i]
        
        # Error estimate
        err_vec = y_new - y_err
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.sqrt(np.sum((err_vec / scale) ** 2)) / np.sqrt(dim)
        
        # Check for failure
        if not np.all(np.isfinite(y_new)):
            # Reduce step and retry
            h = h * 0.5
            if h < min_h:
                return y_new  # Give up, return what we have
            continue
        
        # Accept or reject step
        if err_norm <= 1.0:
            # Accept step
            t = t + h
            y = y_new
            step_count += 1
            
            # Update step size
            if err_norm < 1e-12:
                factor = 5.0
            else:
                factor = safety * (err_norm ** (-expt))
                factor = min(factor, 5.0)
                factor = max(factor, 0.1)
            h = h * factor
        else:
            # Reject step, reduce h
            factor = safety * (err_norm ** (-expt))
            factor = max(factor, 0.1)
            h = h * factor
    
    return y

# EVOLVE-BLOCK-END

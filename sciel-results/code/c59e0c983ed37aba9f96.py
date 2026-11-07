import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    
    # DOPRI5 Coefficients
    # c: nodes
    c = np.array([0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0])
    # A: Butcher tableau
    A = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [44.0/45.0, -56.0/15.0, 32.0/9.0, 0.0, 0.0, 0.0, 0.0],
        [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0, 0.0, 0.0, 0.0],
        [9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0, 0.0, 0.0],
        [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0]
    ])
    # b: weights for 5th order solution
    b = np.array([35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0])
    # b_star: weights for 4th order solution (for error estimation)
    b_star = np.array([51732431.0/575700168.0, 0.0, 75251971.0/160454032.0, 22842748.0/5937528.0, -27752089.0/2688408.0, 21619200.0/13513528.0, 212.0/41000.0])
    
    rtol = 1e-11
    atol = 1e-13
    p = 5 # order of method
    
    # Initial step size estimation
    k1 = f(t0, y)
    norm_y = np.linalg.norm(y)
    norm_k1 = np.linalg.norm(k1)
    
    d = rtol * norm_y + atol
    if norm_k1 > 1e-15:
        h = 0.01 * (d / norm_k1) ** (1.0/(p+1))
    else:
        h = 0.01
        
    # Clamp initial step
    h = np.clip(h, 1e-8, (t1 - t0) / 10.0)
    
    max_steps = 50000
    step_count = 0
    min_h = 1e-12
    max_h = (t1 - t0) / 10.0
    
    while t < t1:
        if t + h > t1:
            h = t1 - t
            
        # Compute stages
        # k[0] is already k1 if first step, but we recompute for generality or reuse?
        # Reusing k1 from previous step is complex due to h change. Just recompute.
        k = np.zeros((6, len(y)))
        k[0] = f(t, y)
        
        for i in range(1, 6):
            # y_stage = y + h * sum(A[i,j] * k[j])
            # A[i, 0:i] are the coefficients
            dot_val = np.dot(A[i, :i], k[:i])
            y_stage = y + h * dot_val
            k[i] = f(t + c[i] * h, y_stage)
            
        # Compute new y (5th order) and y_check (4th order)
        y_new = y + h * np.dot(b, k)
        y_check = y + h * np.dot(b_star, k)
        
        # Error estimation
        # err = y_new - y_check
        # scale = atol + rtol * max(|y|, |y_new|)
        err_vec = y_new - y_check
        scale_vec = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.linalg.norm(err_vec / scale_vec)
        
        if err_norm <= 1.0:
            # Accept step
            t += h
            y = y_new
            # Update step size
            if err_norm > 1e-15:
                factor = 0.9 * (err_norm ** (-0.2))
                factor = np.clip(factor, 0.1, 5.0)
            else:
                factor = 5.0
            h_new = h * factor
            h = np.min(h_new, t1 - t)
            h = np.max(h, min_h)
        else:
            # Reject step
            if err_norm > 1e-15:
                factor = 0.9 * (err_norm ** (-0.25))
                factor = np.clip(factor, 0.1, 1.0)
            else:
                factor = 0.1
            h = h * factor
            h = np.max(h, min_h)
            
        step_count += 1
        if step_count > max_steps:
            break
            
    return y

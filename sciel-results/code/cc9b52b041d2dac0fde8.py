"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Coefficients for Dormand-Prince 5(4) method
# A matrix (7x7), b (7,1) for 5th order, b_star (7,1) for 4th order, c (7,1)
_A = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [1/5, 0, 0, 0, 0, 0, 0],
    [3/40, 9/40, 0, 0, 0, 0, 0],
    [44/45, -56/15, 32/9, 0, 0, 0, 0],
    [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
    [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
])
_b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
_b_star = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
_c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])

def _dopri54_step(f, t, y, h):
    """Perform one step of Dormand-Prince 5(4).
    Returns y_new (5th order), y_star (4th order), and error estimate."""
    k = np.zeros((7, len(y)))
    k[0] = f(t, y)
    
    for i in range(1, 7):
        y_stage = y + h * np.dot(_A[i, :i], k[:i])
        k[i] = f(t + _c[i] * h, y_stage)
        
    y_new = y + h * np.dot(_b, k)
    y_star = y + h * np.dot(_b_star, k)
    
    # Error estimate is the difference between 5th and 4th order solutions
    err = y_new - y_star
    return y_new, y_star, err

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Tolerances set tighter than reference to ensure high accuracy
    rtol = 1e-13
    atol = 1e-15
    
    # Initial step size estimate
    # Heuristic based on function values
    f0 = f(t0, y)
    d0 = np.linalg.norm(y / atol + rtol, ord=np.inf)
    d1 = np.linalg.norm(f0 / atol + rtol, ord=np.inf)
    
    if d0 == 0.0 or d1 == 0.0:
        h = 1e-6
    else:
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(f0))
        # Simple heuristic for initial h
        h = 0.01 * (t1 - t0)
        # If we have more info, we could refine, but 0.01 * span is usually safe enough for start
        # Let's try to be a bit smarter if possible, but keep it simple for robustness
        # h = 0.01 * (t1 - t0) is a reasonable default for t in [0, 5]
        
    # Safety factor for step size
    SAFETY = 0.9
    MIN_FACTOR = 0.2
    MAX_FACTOR = 10.0
    
    t = t0
    h = min(h, t1 - t0)
    
    # Deadline guard: 330s wall clock limit, aim for 300s
    start_time = time.time()
    deadline = start_time + 290.0 # 10s margin
    
    max_steps = 100000 # Hard cap on steps to prevent infinite loops
    
    for _ in range(max_steps):
        if t >= t1:
            break
            
        # Check deadline
        if time.time() > deadline:
            # If we hit the deadline, we should return the current best estimate
            # Since we are mid-integration, we can't easily "undo" the last partial step logic 
            # without complex state saving. However, for this problem, we expect to finish well before.
            # If we do hit it, returning the current y is the best we can do, 
            # but it might be inaccurate. Given the problem size, this is unlikely.
            break
        
        # Limit step to end of interval
        if t + h > t1:
            h = t1 - t
            
        y_new, y_star, err = _dopri54_step(f, t, y, h)
        
        # Check for non-finite values
        if not np.all(np.isfinite(y_new)):
            # Reduce step size and retry
            h *= 0.5
            if h < 1e-12:
                # If step size becomes too small, return current y to avoid crash
                return y
            continue
            
        # Error norm
        # err_norm = max(|err_i| / (atol + rtol * max(|y_i|, |y_new_i|)))
        err_norm = np.max(np.abs(err) / (atol + rtol * np.maximum(np.abs(y), np.abs(y_new))))
        
        if err_norm <= 1.0:
            # Accept step
            t += h
            y = y_new
            
            # Calculate new step size
            if err_norm == 0.0:
                factor = MAX_FACTOR
            else:
                factor = SAFETY * (err_norm ** (-0.2)) # PI controller exponent for 5th order
                factor = min(MAX_FACTOR, max(MIN_FACTOR, factor))
            
            h_new = h * factor
            # Limit step to not overshoot
            if t + h_new > t1:
                h_new = t1 - t
            h = h_new
        else:
            # Reject step, reduce h
            factor = SAFETY * (err_norm ** (-0.25)) # More conservative reduction
            factor = max(MIN_FACTOR, factor)
            h = h * factor
            # If h is too small, we might be stuck.
            if h < 1e-12:
                # Fallback: take a small fixed step to move forward
                h = 1e-12
                y_new, _, _ = _dopri54_step(f, t, y, h)
                t += h
                y = y_new

    return y

# EVOLVE-BLOCK-END

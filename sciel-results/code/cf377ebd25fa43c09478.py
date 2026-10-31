import numpy as np
import time

N = 600

def _autoconv_peak(h):
    """Compute the peak of the autoconvolution of h, normalized by (sum h)^2."""
    s = np.sum(h)
    if s == 0:
        return 0.0
    # Full autoconvolution via FFT for speed
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    H = np.fft.rfft(h, size)
    ac = np.fft.irfft(H * np.conj(H), size)[:2*n-1]
    peak = np.max(ac)
    return peak / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Must finish within 120s
    
    # Start with uniform
    best_h = np.ones(N)
    best_score = _autoconv_peak(best_h)
    
    # Try to improve using iterative optimization
    # Use a simple gradient-free approach: perturb and keep improvements
    
    h = np.ones(N).astype(float)
    current_score = best_score
    
    # Use a local search with random perturbations
    np.random.seed(42)
    
    # Try different strategies within time budget
    # Strategy 1: Small random perturbations
    step_size = 0.01
    max_iters = 50000
    iters = 0
    
    while iters < max_iters:
        if time.time() > deadline - 2.0:
            break
        
        # Random perturbation
        h_new = h.copy()
        idx = np.random.randint(0, N)
        h_new[idx] += np.random.uniform(-step_size, step_size)
        h_new[idx] = max(0.0, h_new[idx])  # Keep nonnegative
        
        score_new = _autoconv_peak(h_new)
        if score_new < current_score:
            h = h_new
            current_score = score_new
            if current_score < best_score:
                best_h = h.copy()
                best_score = current_score
            # Occasionally reduce step size
            step_size *= 0.99
        else:
            step_size *= 1.01
        
        iters += 1
    
    # Ensure nonnegative
    best_h = np.maximum(0.0, best_h)
    
    return [float(x) for x in best_h]

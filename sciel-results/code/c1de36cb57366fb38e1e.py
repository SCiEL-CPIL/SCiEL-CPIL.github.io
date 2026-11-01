"""Optimized signed step function for autoconvolution minimization."""
import numpy as np
import time

N = 400

def _autoconv_fft(x):
    """Compute autoconvolution using FFT."""
    n = len(x)
    size = 2 * n
    f = np.fft.rfft(x, size)
    ac = np.fft.irfft(f * np.conj(f), size)[:n]
    return ac

def _peak_sidelobe(x):
    """Maximum absolute value of autoconvolution at non-zero lags."""
    ac = _autoconv_fft(x)
    if len(ac) <= 1:
        return 0.0
    return np.max(np.abs(ac[1:]))

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    best_x = np.ones(N, dtype=np.float64)
    best_val = _peak_sidelobe(best_x)
    
    rng = np.random.default_rng(789)
    
    def local_search(x, max_iters=10000):
        nonlocal best_val, best_x
        x = x.copy()
        cur_val = _peak_sidelobe(x)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            idx = rng.integers(0, N)
            x_new = x.copy()
            x_new[idx] *= -1
            new_val = _peak_sidelobe(x_new)
            
            if new_val < cur_val:
                x = x_new
                cur_val = new_val
                if cur_val < best_val:
                    best_val = cur_val
                    best_x = x.copy()
        
        return x, cur_val
    
    for restart in range(50):
        if time.time() > deadline:
            break
        if restart == 0:
            x0 = np.ones(N)
        else:
            x0 = rng.choice([-1.0, 1.0], size=N)
        local_search(x0, max_iters=8000)
    
    return best_x

import numpy as np
import time

N = 600
INTERVAL = 1.0 / N  # width of each step on [-1/4, 1/4], total length 1/2

def _autoconv_peak(heights):
    """Compute max of autoconvolution of step function on [-1/4, 1/4]."""
    h = np.asarray(heights, dtype=np.float64)
    # Pad to next power of 2 for FFT efficiency
    n = 2 * N - 1  # length of autoconvolution
    nfft = 1
    while nfft < n:
        nfft *= 2
    H = np.fft.rfft(h, nfft)
    ac = np.fft.irfft(H * H.conj(), nfft)[:n]
    return np.max(ac)

def _score(heights):
    """Raw value: normalized max autoconvolution. Lower is better."""
    h = np.asarray(heights, dtype=np.float64)
    # Normalize so that integral = 1
    integral = np.sum(h) * INTERVAL
    if integral <= 0:
        return float('inf')
    h_norm = h / integral  # now integral = 1
    return _autoconv_peak(h_norm)

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    # Start from uniform
    best = np.ones(N, dtype=np.float64)
    best_val = _score(best)
    
    rng = np.random.default_rng(42)
    
    # Multi-start simulated annealing
    n_starts = 20
    T_init = 0.1
    T_min = 1e-6
    cooling = 0.999
    
    for s in range(n_starts):
        if time.time() > deadline:
            break
        
        # Initialize: either uniform or random
        if s == 0:
            curr = np.ones(N, dtype=np.float64)
        else:
            # Random nonnegative initialization
            curr = rng.random(N, dtype=np.float64)
            if np.sum(curr) <= 0:
                curr = np.ones(N, dtype=np.float64)
        
        curr_val = _score(curr)
        if curr_val < best_val:
            best = curr.copy()
            best_val = curr_val
        
        T = T_init
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline and no_improve < max_no_improve:
            # Perturb: Gaussian noise on a random subset
            n_perturb = rng.integers(5, 50)
            indices = rng.choice(N, size=n_perturb, replace=False)
            noise = rng.normal(0, 0.1, size=n_perturb)
            new_h = curr.copy()
            new_h[indices] += noise
            new_h = np.maximum(new_h, 0.0)  # enforce nonnegativity
            
            new_val = _score(new_h)
            
            if new_val < curr_val:
                curr = new_h
                curr_val = new_val
                no_improve = 0
                if curr_val < best_val:
                    best = curr.copy()
                    best_val = curr_val
            else:
                # Accept with probability exp(-(new_val - curr_val)/T)
                delta = new_val - curr_val
                if delta > 0 and T > T_min:
                    if rng.random() < np.exp(-delta / T):
                        curr = new_h
                        curr_val = new_val
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            T *= cooling
            if T < T_min:
                T = T_min
    
    # Return as list of floats
    return [float(x) for x in best]

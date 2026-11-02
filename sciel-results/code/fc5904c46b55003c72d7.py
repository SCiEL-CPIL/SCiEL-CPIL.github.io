"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(x):
    """Compute max absolute value of linear autoconvolution of x."""
    # Use FFT for efficient convolution
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    f = np.fft.rfft(x, size)
    ac = np.fft.irfft(f * np.conj(f), size)[:2*n-1]
    return np.max(np.abs(ac))

def _score(x):
    """Score = max_abs_autoconv / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if s == 0:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def construct():
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s limit
    
    best_x = None
    best_val = float('inf')
    
    # Start with random signed sequence
    rng = np.random.default_rng(42)
    x = rng.choice([-1.0, 1.0], size=N)
    
    # Try multiple random initializations and local search
    n_attempts = 20
    
    for attempt in range(n_attempts):
        if time.time() - start_time > deadline - 5:
            break
        
        # Random signed sequence
        x = rng.choice([-1.0, 1.0], size=N)
        val = _score(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        # Local search: flip individual elements to reduce score
        improved = True
        while improved and (time.time() - start_time < deadline - 2):
            improved = False
            # Try flipping each element
            for i in range(N):
                if time.time() - start_time > deadline - 2:
                    break
                old_val = x[i]
                x[i] = -old_val
                new_val = _score(x)
                if new_val < val - 1e-12:
                    val = new_val
                    improved = True
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                else:
                    x[i] = old_val
    
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(f):
    """Compute max absolute value of autoconvolution (excluding self-lag 0)."""
    # Use FFT for efficient convolution
    n = len(f)
    # Zero-pad to avoid circular convolution
    size = 2 * n
    F = np.fft.fft(f, n=size)
    conv = np.fft.ifft(F * F).real[:n]
    # conv[0] is the self-convolution at lag 0 = sum(f^2) = n for ±1
    # We want max |conv[k]| for k != 0, but also consider the bound includes all
    # The "upper bound on C3" likely refers to max over all lags
    return np.max(np.abs(conv))


def _try_flip(f, i):
    """Try flipping position i and return new score."""
    f_new = f.copy()
    f_new[i] *= -1.0
    return _autoconv_max_abs(f_new), f_new


def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    rng = np.random.RandomState(42)
    
    # Initialize with random ±1 values
    best_f = rng.choice([-1.0, 1.0], size=N)
    best_score = _autoconv_max_abs(best_f)
    
    # Local search with simulated annealing style
    T = 50.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < deadline:
        # Pick a random position to flip
        i = rng.randint(N)
        
        f_new = best_f.copy()
        f_new[i] *= -1.0
        new_score = _autoconv_max_abs(f_new)
        
        if new_score < best_score or (T > T_min and rng.rand() < np.exp(-(new_score - best_score) / T)):
            best_f = f_new
            best_score = new_score
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_f
# EVOLVE-BLOCK-END

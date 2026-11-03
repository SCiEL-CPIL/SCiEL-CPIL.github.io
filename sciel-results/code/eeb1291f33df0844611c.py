"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_peak_sidelobe(s):
    """Compute the maximum absolute value of the autoconvolution sidelobes (lags 1..N-1)."""
    # Use FFT for efficient autocorrelation
    n = len(s)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    f = np.fft.rfft(s, n=size)
    acf = np.fft.irfft(f * np.conj(f), n=size)[:n]
    # Sidelobes are lags 1 to N-1
    if n > 1:
        return np.max(np.abs(acf[1:]))
    return 0.0


def _autoconv_peak_sidelobe_fast(s):
    """Faster version using direct computation for moderate N, or FFT."""
    n = len(s)
    if n <= 512:
        # Direct computation might be faster for small n due to FFT overhead
        # But for n=400, FFT should be fine
        size = 1
        while size < 2 * n - 1:
            size *= 2
        f = np.fft.rfft(s, n=size)
        acf = np.fft.irfft(f * np.conj(f), n=size)[:n]
        if n > 1:
            return np.max(np.abs(acf[1:]))
        return 0.0
    else:
        return _autoconv_peak_sidelobe(s)


def construct():
    """Construct a signed step function to minimize the autoconvolution peak sidelobe."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initialize with random signs
    np.random.seed(42)
    s = np.random.choice([-1.0, 1.0], size=N)
    
    # Evaluate initial
    best_s = s.copy()
    best_val = _autoconv_peak_sidelobe_fast(s)
    
    # Simulated annealing with block flips
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.999
    max_block_size = 32
    min_block_size = 4
    
    current_val = best_val
    current_s = s.copy()
    
    iteration = 0
    while T > T_min:
        if time.time() > deadline - 5.0:
            break
        
        # Choose block size
        block_size = np.random.randint(min_block_size, max_block_size + 1)
        
        # Choose random start position
        if block_size >= N:
            start = 0
            block_size = N
        else:
            start = np.random.randint(0, N - block_size + 1)
        
        # Create proposed flip
        proposed_s = current_s.copy()
        proposed_s[start:start+block_size] *= -1.0
        
        # Evaluate proposed
        proposed_val = _autoconv_peak_sidelobe_fast(proposed_s)
        
        # Acceptance criterion
        delta = proposed_val - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_s = proposed_s
            current_val = proposed_val
            if current_val < best_val:
                best_val = current_val
                best_s = current_s.copy()
        
        # Cool down
        T *= cooling_rate
        iteration += 1
        
        # Periodically check time
        if iteration % 100 == 0 and time.time() > deadline - 10.0:
            break
    
    # Final polish: single-bit flips near the end
    if time.time() < deadline - 5.0:
        T_polish = 0.01
        for _ in range(5000):
            if time.time() > deadline - 3.0:
                break
            idx = np.random.randint(0, N)
            proposed_s = best_s.copy()
            proposed_s[idx] *= -1.0
            proposed_val = _autoconv_peak_sidelobe_fast(proposed_s)
            if proposed_val < best_val:
                best_val = proposed_val
                best_s = proposed_s.copy()
    
    return best_s
# EVOLVE-BLOCK-END

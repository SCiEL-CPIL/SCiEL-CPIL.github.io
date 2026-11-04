"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_peak_autocorr(h):
    """Compute the maximum absolute value of the linear autocorrelation at non-zero lags."""
    n = len(h)
    # Linear autocorrelation: r[k] = sum_i h[i] * h[i-k] for k > 0
    # Use FFT for efficiency
    fft_size = 1
    while fft_size < 2 * n - 1:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    # Power spectrum
    power = np.abs(H) ** 2
    # Inverse FFT gives circular autocorrelation
    auto = np.fft.irfft(power, n=fft_size)
    # The first n values are r[0], r[1], ..., r[n-1]
    # r[0] is the DC term (sum of squares), exclude it
    # We want max |r[k]| for k = 1, ..., n-1
    # Also need to check negative lags, but for real sequences r[-k] = r[k]
    peak = np.max(np.abs(auto[1:n]))
    return peak


def _local_search(h, time_limit):
    """Local search: flip individual bits to reduce peak autocorrelation."""
    n = len(h)
    h = h.copy()
    current_peak = _compute_peak_autocorr(h)
    best_peak = current_peak
    best_h = h.copy()
    
    # Precompute phases for fast FFT update
    # Actually, for peak autocorrelation we need to recompute, so let's use a different approach
    # Use greedy: try each flip, keep the best one, repeat
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try flipping each position
        for i in range(n):
            h[i] = -h[i]
            new_peak = _compute_peak_autocorr(h)
            if new_peak < current_peak - 1e-12:
                current_peak = new_peak
                improved = True
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_h = h.copy()
                no_improve_count = 0
                break
            else:
                h[i] = -h[i]  # revert
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
    
    return best_h, best_peak


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    rng = np.random.RandomState(42)
    
    best_h = None
    best_peak = np.inf
    
    # Multiple random restarts
    n_restarts = 20
    time_per_restart = (deadline - start_time) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() >= deadline:
            break
        
        # Initialize with random signs
        h = rng.choice([-1.0, 1.0], size=N)
        
        # Run local search with time limit
        remaining = deadline - time.time() - 5.0  # 5s buffer
        if remaining <= 0:
            break
        time_limit = min(remaining, time_per_restart)
        
        h_opt, peak = _local_search(h, time_limit)
        
        if peak < best_peak:
            best_peak = peak
            best_h = h_opt
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END

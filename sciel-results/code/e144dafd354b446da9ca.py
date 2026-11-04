"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_peak(f):
    """Compute peak absolute autocorrelation at non-zero lags."""
    n = len(f)
    # Use FFT for efficient autocorrelation
    f_float = f.astype(np.float64)
    # Zero-pad to avoid circular convolution issues
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.fft(f_float, size)
    acf = np.fft.ifft(F * np.conj(F)).real[:n]
    # Peak at non-zero lags
    peak = np.max(np.abs(acf[1:]))
    return peak


def _local_search(f, max_time=100.0):
    """Local search to minimize peak autocorrelation."""
    start_time = time.time()
    best_peak = _autocorr_peak(f)
    best_f = f.copy()
    n = len(f)
    
    # Precompute autocorrelation for faster updates
    # When we flip position i, the change in autocorrelation at lag k is:
    # For k > 0: delta_acf[k] = -2*f[i] * (f[i+k] + f[i-k]) if both in range
    # This is complex, so let's use a simpler approach with periodic checks
    
    rng = np.random.default_rng(42)
    stall_count = 0
    best_stall = 0
    
    while time.time() - start_time < max_time:
        # Randomly pick a position to flip
        i = rng.integers(0, n)
        f[i] = -f[i]
        new_peak = _autocorr_peak(f)
        
        if new_peak < best_peak:
            best_peak = new_peak
            best_f = f.copy()
            stall_count = 0
        else:
            # Revert if not improving
            f[i] = -f[i]
            stall_count += 1
            best_stall = max(best_stall, stall_count)
        
        # Adaptive step: if stuck for a while, try more random restarts
        if stall_count > 500:
            # Random restart with some perturbation
            for _ in range(10):
                j = rng.integers(0, n)
                f[j] = -f[j]
            new_peak = _autocorr_peak(f)
            if new_peak < best_peak:
                best_peak = new_peak
                best_f = f.copy()
                stall_count = 0
            else:
                # Revert the 10 flips
                for _ in range(10):
                    j = rng.integers(0, n)  # This won't work, need to track
                    pass
                # Just reset to best
                f = best_f.copy()
                stall_count = 0
    
    return best_f, best_peak


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Start with random ±1 sequence
    rng = np.random.default_rng(12345)
    f = rng.choice([-1, 1], size=N)
    
    best_f, best_peak = _local_search(f, max_time=100.0)
    
    # Check if we're still within deadline
    if time.time() > deadline:
        return best_f
    
    # Try a few more random restarts if time permits
    for seed in [99, 77, 55]:
        if time.time() > deadline:
            break
        f = rng.choice([-1, 1], size=N)
        f, peak = _local_search(f, max_time=min(20.0, deadline - time.time()))
        if peak < best_peak:
            best_f = f
            best_peak = peak
    
    return best_f
# EVOLVE-BLOCK-END

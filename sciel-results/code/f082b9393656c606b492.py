import numpy as np
import time

N = 400
TIME_LIMIT = 145  # safety margin before 150s kill

def _compute_c3_bound(heights):
    """
    Compute the upper bound on C3 from the autoconvolution of the signed step function.
    Returns dx * max_k |autocorr[k]| where autocorr is the aperiodic autocorrelation.
    """
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    dx = 0.5 / n
    
    # Compute aperiodic autocorrelation via FFT
    # Pad to avoid circular convolution
    size = 1
    while size < 2 * n - 1:
        size *= 2
    size = max(size, 2 * n - 1)
    
    h_pad = np.zeros(size)
    h_pad[:n] = h
    
    F = np.fft.fft(h_pad)
    power = np.abs(F) ** 2
    autocorr = np.fft.ifft(power).real
    
    # The aperiodic autocorrelation at lag k for k=0..n-1 is autocorr[k]
    # For negative lags, it's autocorr[-k % size] = autocorr[size-k]
    # But for a real sequence, autocorr[k] = autocorr[-k], so we just need 0..n-1
    max_abs = np.max(np.abs(autocorr[:n]))
    
    c3_bound = dx * max_abs
    return c3_bound


def _compute_autocorr(h):
    """Compute aperiodic autocorrelation efficiently."""
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    size = max(size, 2 * n - 1)
    
    h_pad = np.zeros(size)
    h_pad[:n] = h
    
    F = np.fft.fft(h_pad)
    power = np.abs(F) ** 2
    autocorr = np.fft.ifft(power).real[:n]
    return autocorr


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n (as +/-1)."""
    tm = np.zeros(n, dtype=np.float64)
    tm[0] = 1.0
    for i in range(1, n):
        # Count trailing zeros in (i-1), or use bit count
        tm[i] = tm[i >> 1] * (-1.0 if (i & 1) else 1.0)
    # Alternative: use popcount
    tm2 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # Popcount of i
        pc = bin(i).count('1')
        tm2[i] = 1.0 if pc % 2 == 0 else -1.0
    return tm2


def _local_search_fast(h, max_iter=200, time_budget=30.0):
    """
    Fast local search: flip individual signs to reduce max |autocorr|.
    Uses the fact that flipping h[i] changes autocorrelation in a predictable way.
    """
    h = np.array(h, dtype=np.float64)
    n = len(h)
    current_bound = _compute_c3_bound(h)
    
    start = time.time()
    improved = True
    iters = 0
    
    while improved and iters < max_iter and (time.time() - start) < time_budget:
        improved = False
        iters += 1
        
        # Compute current autocorrelation
        ac = _compute_autocorr(h)
        current_max = np.max(np.abs(ac))
        
        best_gain = 0
        best_idx = -1
        
        # Try flipping each position
        for i in range(n):
            # Flipping h[i] from h[i] to -h[i]
            # The autocorrelation change: 
            # new_ac[k] = ac[k] - 2*h[i]*sum_j h[j]*h[j+k] for the terms involving i
            # This is complex, so let's use a simpler approach:
            # Just recompute for the top candidates
            
            # For efficiency, estimate the gain by checking how much h[i] contributes
            # to the peaks
            pass
        
        # Simpler: try a subset of positions (those with high contribution to peaks)
        # Find positions that contribute most to the max autocorrelation
        peak_lag = np.argmax(np.abs(ac))
        
        # Try flipping positions in a random order, keep first improvement
        order = np.random.permutation(n)
        found = False
        for i in order:
            if (time.time() - start) > time_budget:
                break
            old_val = h[i]
            h[i] = -old_val
            new_bound = _compute_c3_bound(h)
            h[i] = old_val
            
            gain = current_bound - new_bound
            if gain > 1e-12:
                h[i] = -h[i]
                current_bound = new_bound
                improved = True
                found = True
                break
        
        if not found:
            improved = False
    
    return h, current_bound


def _simulated_annealing(h, time_budget=20.0):
    """
    Simulated annealing to escape local minima.
    """
    h = np.array(h, dtype=np.float64)
    n = len(h)
    current_bound = _compute_c3_bound(h)
    best_h = h.copy()
    best_bound = current_bound
    
    start = time.time()
    T = 0.1  # initial temperature
    cooling = 0.99
    
    while (time.time() - start) < time_budget:
        # Pick a random position to flip
        i = np.random.randint(n)
        old_val = h[i]
        h[i] = -old_val
        new_bound = _compute_c3_bound(h)
        
        delta = new_bound - current_bound
        if delta < 0 or np.random.rand() < np.exp(-delta / (T + 1e-10)):
            current_bound = new_bound
            if new_bound < best_bound:
                best_bound = new_bound
                best_h = h.copy()
        else:
            h[i] = old_val
        
        T *= cooling
    
    return best_h, best_bound


def construct():
    start_time = time.time()
    
    # Initialize with Thue-Morse sequence (known low autocorrelation)
    best_h = _thue_morse(N)
    best_bound = _compute_c3_bound(best_h)
    
    # Try a few different initializations
    np.random.seed(12345)
    
    # Also try random initializations
    for trial in range(10):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        h = np.random.choice([-1.0, 1.0], size=N)
        bound = _compute_c3_bound(h)
        if bound < best_bound:
            best_bound = bound
            best_h = h.copy()
    
    # Local search from best initialization
    if time.time() - start_time < TIME_LIMIT - 20:
        best_h, best_bound = _local_search_fast(best_h, max_iter=100, time_budget=20.0)
    
    # Simulated annealing if time permits
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 25:
        sa_h, sa_bound = _simulated_annealing(best_h, time_budget=min(20.0, remaining - 10))
        if sa_bound < best_bound:
            best_bound = sa_bound
            best_h = sa_h.copy()
            # Do local search again from SA result
            if time.time() - start_time < TIME_LIMIT - 10:
                best_h, best_bound = _local_search_fast(best_h, max_iter=50, time_budget=10.0)
    
    # Additional restarts if time permits
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 20:
        for trial in range(5):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            h = np.random.choice([-1.0, 1.0], size=N)
            bound = _compute_c3_bound(h)
            if bound < best_bound:
                h, bound = _local_search_fast(h, max_iter=50, time_budget=10.0)
                if bound < best_bound:
                    best_bound = bound
                    best_h = h.copy()
    
    # Final local search polish
    if time.time() - start_time < TIME_LIMIT - 5:
        best_h, best_bound = _local_search_fast(best_h, max_iter=30, time_budget=5.0)
    
    return best_h


if __name__ == "__main__":
    h = construct()
    print(f"Length: {len(h)}")
    print(f"First 20: {h[:20]}")
    print(f"C3 bound: {_compute_c3_bound(h)}")
    print(f"Score: {1.0 / _compute_c3_bound(h)}")

"""
Third autocorrelation inequality: minimize the upper bound on C3.
The score is seed/value where value is the raw upper bound (lower is better).
We search over signed step functions with 400 intervals on [-1/4, 1/4].
"""
import numpy as np
import time

N = 400
TIME_LIMIT = 140  # safety margin before 150s kill

def _compute_c3_bound(heights):
    """
    Compute the upper bound on C3 from the autoconvolution of the signed step function.
    
    For a step function f(x) = sum_i h_i * 1_{[x_i, x_{i+1}]}(x) on [-1/4, 1/4],
    the autoconvolution (f * f)(t) = integral f(s) f(t-s) ds.
    
    The C3 bound relates to the L-infinity norm of the autoconvolution.
    Specifically, for the third autocorrelation inequality, the bound is:
        C3 <= max_t |(f*f)(t)|
    
    where f is the signed step function. We compute this via FFT for efficiency.
    """
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    # Interval width
    dx = 0.5 / n  # total length 0.5, n intervals
    
    # Pad h to length 2n-1 for linear convolution
    h_pad = np.zeros(2 * n - 1)
    h_pad[:n] = h
    
    # Linear autoconvolution via FFT
    f_fft = np.fft.fft(h_pad, n=2 * (2 * n - 1))
    conv = np.fft.ifft(f_fft * np.conj(f_fft)).real
    
    # Compute max absolute value
    max_abs = np.max(np.abs(conv))
    c3_bound = dx * max_abs
    
    return c3_bound

def _thue_morse_sequence(n):
    """Generate Thue-Morse sequence of length n."""
    result = np.zeros(n)
    for i in range(n):
        # Count bits in i
        count = 0
        x = i
        while x:
            count += x & 1
            x >>= 1
        result[i] = 1.0 if count % 2 == 0 else -1.0
    return result

def _local_search(heights, time_budget=100.0):
    """
    Local search: try flipping individual bits (changing sign) to reduce the C3 bound.
    Uses time-based stopping.
    """
    h = np.array(heights, dtype=np.float64)
    current_bound = _compute_c3_bound(h)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_budget:
        improved = False
        best_gain = 0
        best_idx = -1
        
        for i in range(N):
            if time.time() - start_time > time_budget:
                break
            old_val = h[i]
            h[i] = -old_val
            new_bound = _compute_c3_bound(h)
            h[i] = old_val
            
            gain = current_bound - new_bound
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx >= 0:
            h[best_idx] = -h[best_idx]
            current_bound -= best_gain
            improved = True
    
    return h, current_bound

def construct():
    start_time = time.time()
    
    # Initialize with Thue-Morse
    best_h = _thue_morse_sequence(N)
    best_bound = _compute_c3_bound(best_h)
    
    # Local search from Thue-Morse initialization
    if time.time() - start_time < TIME_LIMIT - 15:
        best_h, best_bound = _local_search(best_h, time_budget=TIME_LIMIT - 20)
    
    # Additional restarts with random signs if time permits
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 20:
        np.random.seed(42)
        for trial in range(10):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            h = np.random.choice([-1.0, 1.0], size=N)
            bound = _compute_c3_bound(h)
            if bound < best_bound:
                h, bound = _local_search(h, time_budget=max(5, TIME_LIMIT - (time.time() - start_time) - 5))
                if bound < best_bound:
                    best_bound = bound
                    best_h = h.copy()
    
    return best_h

# Test
if __name__ == "__main__":
    h = construct()
    print(f"Length: {len(h)}")
    print(f"First 20: {h[:20]}")
    print(f"C3 bound: {_compute_c3_bound(h)}")
    print(f"Score: {1.0 / _compute_c3_bound(h)}")

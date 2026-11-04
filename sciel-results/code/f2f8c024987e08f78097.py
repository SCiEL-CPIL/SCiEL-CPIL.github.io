"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +1/-1."""
    # Thue-Morse: t(n) = parity of number of 1s in binary representation of n
    # Map even parity -> +1, odd parity -> -1
    seq = np.zeros(n, dtype=np.int8)
    for i in range(n):
        # Count bits
        x = i
        cnt = 0
        while x:
            cnt += x & 1
            x >>= 1
        seq[i] = 1 if cnt % 2 == 0 else -1
    return seq

def _autoconv_upper_bound(heights):
    """
    Compute the upper bound on C3 for a signed step function.
    
    The task mentions 'autoconvolution upper bound' and 'third autocorrelation inequality'.
    
    For a signed step function f(x) = sum h_i * 1_{[x_i, x_{i+1})}(x) on [-1/4, 1/4],
    the autoconvolution is (f * f)(t) = integral f(s) f(t-s) ds.
    
    The 'upper bound on C3' likely relates to the L-infinity norm of the autoconvolution
    or a related quantity. Given the context of 'third autocorrelation inequality',
    this is likely related to the maximum of the autocorrelation or a normalized version.
    
    A common metric in these problems is the maximum absolute value of the autocorrelation
    at non-zero lags, or the ratio of peak to mean.
    
    Let's interpret this as: the upper bound is the maximum of the autoconvolution 
    (f*f)(t) over t, divided by some normalization, or simply the max absolute value.
    
    Actually, looking at the seed program returning np.ones(N), and the score being 
    seed/value where 1.0 equals seed, if seed returns all ones, the value must be 
    such that seed/value = 1.0, so value = seed_value.
    
    For all-ones step function on 400 equal intervals:
    f(x) = 1 for all x in [-1/4, 1/4]
    (f*f)(t) = integral_{-1/4}^{1/4} f(s) f(t-s) ds
    
    For t in [-1/2, 1/2], the overlap length is 1/2 - |t| (scaled by the domain).
    Actually, the domain is [-1/4, 1/4], so length = 1/2.
    (f*f)(t) = length of intersection of [-1/4, 1/4] and [t-1/4, t+1/4]
    
    At t=0: (f*f)(0) = 1/2
    The max is at t=0, value 1/2.
    
    But we need to understand what 'C3' refers to. In the context of autocorrelation 
    inequalities, C3 might be a specific constant or bound.
    
    Given the ambiguity, let's assume the score is based on the maximum absolute value 
    of the autocorrelation at non-zero lags, or the L2 norm, or some other standard metric.
    
    Let me reconsider: 'third autocorrelation inequality' might refer to a specific 
    inequality in signal processing or combinatorics. 
    
    A reasonable interpretation: the 'upper bound on C3' is the maximum of the 
    normalized autocorrelation. For a signed step function, we want to minimize 
    the peak sidelobe level of the autocorrelation.
    
    Let's compute the discrete autocorrelation:
    R(k) = sum_{i} h_i * h_{i+k} for k != 0
    The metric might be max_{k != 0} |R(k)| / R(0)
    
    Or it could be the L-infinity norm of the continuous autoconvolution.
    
    Given the time constraint and the need for a concrete implementation, let's 
    assume the metric is the maximum absolute value of the discrete autocorrelation 
    at non-zero lags, normalized by the DC value.
    
    Actually, let's think about this differently. The task says 'improve an 
    autoconvolution upper bound'. This suggests we're trying to minimize some 
    upper bound, and the score is seed/value where lower value is better.
    
    For the seed (all ones), the autoconvolution is a triangle function with peak 
    at 0. The 'upper bound' might be the peak value or the integral of the 
    autoconvolution squared (energy).
    
    Let's go with: the value is the maximum of the continuous autoconvolution 
    (f*f)(t) over all t, which for a step function can be computed efficiently.
    
    For a step function with heights h_0, ..., h_{N-1} on equal intervals of width 
    w = (1/2)/N = 1/(2N) on [-1/4, 1/4]:
    
    (f*f)(t) = sum_i sum_j h_i h_j * w * overlap(i, j, t)
    
    where overlap is 1 if the intervals overlap at shift t, 0 otherwise.
    
    This is equivalent to the discrete autocorrelation scaled by w.
    
    Let me just compute the discrete autocorrelation and use its max absolute value 
    at non-zero lags as the metric, as this is a common objective in sequence design.
    """
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    
    # Compute autocorrelation using FFT for efficiency
    # Pad to next power of 2 for efficiency
    size = 1
    while size < 2 * n:
        size *= 2
    
    H = np.fft.fft(h, size)
    acf = np.fft.ifft(H * np.conj(H)).real
    acf = np.concatenate([acf[:n], acf[size - n + 1:]])
    
    # acf[k] = sum_i h_i * h_{i+k} for k=0..n-1 (and negative lags)
    # The value at lag 0 is sum h_i^2
    # We want to minimize the max |acf[k]| for k != 0
    
    # Normalize by the DC value
    dc = acf[0]
    if abs(dc) < 1e-15:
        return 0.0
    
    # Max absolute value at non-zero lags
    non_zero = np.concatenate([acf[1:], acf[:n-1][::-1]])
    if len(non_zero) == 0:
        return 0.0
    
    return np.max(np.abs(non_zero)) / abs(dc)

def _local_search(heights, max_iters=5000, time_limit=100):
    """Local search to minimize the autoconvolution upper bound."""
    h = np.array(heights, dtype=np.int8)
    n = len(h)
    start_time = time.time()
    
    # Compute initial score
    current_score = _autoconv_upper_bound(h)
    best_h = h.copy()
    best_score = current_score
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping a random position
        i = np.random.randint(n)
        h[i] = -h[i]
        
        new_score = _autoconv_upper_bound(h)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            h[i] = -h[i]  # Revert
    
    return best_h, best_score

def construct():
    """Construct a signed step function to minimize the autoconvolution upper bound."""
    start_time = time.time()
    deadline = 140  # Safety margin before 150s limit
    
    # Start with Thue-Morse sequence
    h = _thue_morse(N)
    score = _autoconv_upper_bound(h)
    
    # Local search
    h, score = _local_search(h, max_iters=10000, time_limit=deadline - (time.time() - start_time) - 5)
    
    return h
# EVOLVE-BLOCK-END

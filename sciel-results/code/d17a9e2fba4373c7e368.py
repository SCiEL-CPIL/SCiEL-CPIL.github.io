"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

TIME_LIMIT = 145  # safety margin before 150s kill


def _compute_c3_bound(heights):
    """
    Compute the upper bound on C3 from the autoconvolution of the signed step function.
    For a step function f(x) with N equal intervals on [-1/4, 1/4],
    the autoconvolution (f*f)(t) is computed via FFT-based convolution.
    The bound is dx * max_k |conv[k]| where conv is the discrete autoconvolution.
    """
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    dx = 0.5 / n  # total length 0.5, n intervals
    
    # Pad h for linear autoconvolution
    h_pad = np.zeros(2 * n - 1)
    h_pad[:n] = h
    
    # Linear autoconvolution via FFT
    f_fft = np.fft.fft(h_pad, n=2 * (2 * n - 1))
    conv = np.fft.ifft(f_fft * np.conj(f_fft)).real
    
    # The continuous autoconvolution at grid points is dx * conv[k]
    max_abs = np.max(np.abs(conv))
    c3_bound = dx * max_abs
    
    return c3_bound


def _haar_wavelet_init(n, seed, beta=0.5, num_scales=None):
    """
    Generate a signed step function using Haar wavelet transform with 
    frequency-dependent variance to suppress high-frequency autoconvolution peaks.
    
    The idea: apply inverse Haar wavelet transform to Gaussian noise where
    the variance decreases with frequency (scale), which suppresses high-frequency
    components that tend to create large autoconvolution peaks.
    """
    rng = np.random.RandomState(seed)
    
    if num_scales is None:
        num_scales = int(np.floor(np.log2(n)))
    
    # Generate Haar wavelet coefficients with frequency-dependent variance
    # In the Haar basis, coefficient at scale j (j=0 is coarsest) has variance sigma_j^2
    # We want to suppress high frequencies (fine scales), so sigma_j decreases with j
    coeffs = np.zeros(n)
    
    # Haar wavelet transform structure:
    # Scale 0: 1 coefficient (mean)
    # Scale 1: 2 coefficients
    # Scale j: 2^j coefficients
    # Total: sum_{j=0}^{log2(n)-1} 2^j = n-1, plus 1 for the mean = n
    
    for j in range(num_scales):
        num_coeffs = 2 ** j
        # Variance decreases with scale (frequency)
        # beta controls how aggressively we suppress high frequencies
        variance = (1.0 / (1.0 + beta * j)) ** 2
        coeffs[j * num_coeffs : (j + 1) * num_coeffs] = rng.randn(num_coeffs) * np.sqrt(variance)
    
    # The mean coefficient (coarsest scale)
    coeffs[0] = rng.randn() * 1.0
    
    # Apply inverse Haar wavelet transform
    # We need to implement the inverse Haar transform
    h = _inverse_haar(coeffs, n)
    
    # Project to ±1 hypercube
    h = np.sign(h)
    h[h == 0] = 1.0  # Handle zeros by assigning +1
    
    return h


def _inverse_haar(coeffs, n):
    """
    Apply inverse Haar wavelet transform to get the signal.
    coeffs is arranged as: [mean, scale1_coeffs, scale2_coeffs, ...]
    """
    # Start with the mean
    signal = np.zeros(n)
    signal[:] = coeffs[0]
    
    # For each scale, add the detail coefficients
    offset = 1
    for j in range(int(np.floor(np.log2(n)))):
        num_coeffs = 2 ** j
        detail = coeffs[offset : offset + num_coeffs]
        offset += num_coeffs
        
        # Add detail coefficients to the signal
        # In Haar transform, detail at position i in scale j affects
        # positions 2*i and 2*i+1 in the signal
        for i in range(num_coeffs):
            pos = i * (n // num_coeffs)
            if pos < n:
                signal[pos] += detail[i]
                if pos + 1 < n:
                    signal[pos + 1] -= detail[i]
    
    return signal


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1 values."""
    seq = [1]
    while len(seq) < n:
        seq += [-x for x in seq]
    return np.array(seq[:n], dtype=np.float64)


def _chaotic_sequence(n, r=3.99, x0=0.12345, seed=42):
    """
    Generate a signed sequence from the logistic map near the edge of chaos.
    """
    x = x0
    seq = np.empty(n, dtype=np.float64)
    for i in range(n):
        seq[i] = 1.0 if x > 0.5 else -1.0
        x = r * x * (1.0 - x)
        if x < 0:
            x = 0.0
        elif x > 1:
            x = 1.0
    return seq


def _local_search(h, max_iter=2000, time_budget=10):
    """
    Local search: try flipping individual bits (changing sign) to reduce the C3 bound.
    Uses first-improvement strategy with time-based stopping.
    """
    h = np.array(h, dtype=np.float64)
    current_bound = _compute_c3_bound(h)
    start_time = time.time()
    
    improved = True
    iters = 0
    while improved and iters < max_iter:
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        iters += 1
        best_gain = 0
        best_idx = -1
        
        # Try flipping each position
        for i in range(N):
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


def _pairwise_flip_search(h, max_iter=1000, time_budget=10):
    """
    Enhanced local search: try flipping pairs of bits to escape local minima.
    """
    h = np.array(h, dtype=np.float64)
    current_bound = _compute_c3_bound(h)
    start_time = time.time()
    
    improved = True
    iters = 0
    while improved and iters < max_iter:
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        iters += 1
        best_gain = 0
        best_pair = None
        
        # Try flipping pairs (i, j) where i < j within a window
        window = 50
        for i in range(N):
            if time.time() - start_time > time_budget:
                break
            for j in range(i + 1, min(i + window, N)):
                old_i, old_j = h[i], h[j]
                h[i] = -old_i
                h[j] = -old_j
                new_bound = _compute_c3_bound(h)
                h[i] = old_i
                h[j] = old_j
                
                gain = current_bound - new_bound
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (i, j)
        
        if best_pair is not None:
            i, j = best_pair
            h[i] = -h[i]
            h[j] = -h[j]
            current_bound -= best_gain
            improved = True
    
    return h, current_bound


def construct():
    start_time = time.time()
    
    # Initialize with multiple starting points
    best_h = np.ones(N)
    best_bound = _compute_c3_bound(best_h)
    
    # Try Thue-Morse sequence
    tm = _thue_morse(N)
    tm_bound = _compute_c3_bound(tm)
    if tm_bound < best_bound:
        best_bound = tm_bound
        best_h = tm.copy()
    
    # Try Haar wavelet-based initializations with different beta values
    beta_values = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    for beta in beta_values:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        for seed in [42, 123, 456, 789]:
            if time.time() - start_time > TIME_LIMIT - 20:
                break
            h_wavelet = _haar_wavelet_init(N, seed=seed, beta=beta)
            bound = _compute_c3_bound(h_wavelet)
            if bound < best_bound:
                best_bound = bound
                best_h = h_wavelet.copy()
    
    # Try chaotic sequences with various parameters
    chaotic_params = [
        (3.99, 0.12345),
        (3.999, 0.12345),
        (3.99, 0.45678),
        (3.999, 0.45678),
    ]
    for r, x0 in chaotic_params:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        h_chaos = _chaotic_sequence(N, r=r, x0=x0)
        bound = _compute_c3_bound(h_chaos)
        if bound < best_bound:
            best_bound = bound
            best_h = h_chaos.copy()
    
    # Try random initializations
    np.random.seed(42)
    for trial in range(10):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        h = np.random.choice([-1.0, 1.0], size=N)
        bound = _compute_c3_bound(h)
        if bound < best_bound:
            best_bound = bound
            best_h = h.copy()
    
    # Local search from best initialization
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 15:
        local_time = min(remaining - 10, 50)
        best_h, best_bound = _local_search(best_h, max_iter=5000, time_budget=local_time)
    
    # Pairwise flip search to escape local minima
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 20:
        pair_time = min(remaining - 15, 30)
        best_h, best_bound = _pairwise_flip_search(best_h, max_iter=2000, time_budget=pair_time)
    
    # Additional restarts if time permits
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 25:
        for trial in range(5):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            # Mix of random, chaotic, and wavelet initializations
            if trial % 3 == 0:
                h = np.random.choice([-1.0, 1.0], size=N)
            elif trial % 3 == 1:
                h = _chaotic_sequence(N, r=3.99 + 0.001 * trial, x0=0.1 + 0.01 * trial)
            else:
                h = _haar_wavelet_init(N, seed=1000 + trial, beta=0.5 + 0.3 * trial)
            
            bound = _compute_c3_bound(h)
            
            if bound < best_bound:
                local_time = min(TIME_LIMIT - (time.time() - start_time) - 5, 15)
                h, bound = _local_search(h, max_iter=2000, time_budget=local_time)
                if bound < best_bound:
                    best_bound = bound
                    best_h = h.copy()
    
    # Final local polish
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 10:
        polish_time = min(remaining - 5, 15)
        best_h, best_bound = _local_search(best_h, max_iter=3000, time_budget=polish_time)
    
    # Final verification
    final_bound = _compute_c3_bound(best_h)
    
    return best_h


# Test
if __name__ == "__main__":
    h = construct()
    print(f"Length: {len(h)}")
    print(f"First 20: {h[:20]}")
    print(f"C3 bound: {_compute_c3_bound(h)}")
    print(f"Score: {1.0 / _compute_c3_bound(h)}")
# EVOLVE-BLOCK-END

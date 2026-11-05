"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

TIME_LIMIT = 145  # safety margin before 150s kill


def _compute_convolution(h):
    """
    Compute the linear autoconvolution of h via FFT.
    Returns the convolution array of length 2*n-1.
    """
    n = len(h)
    h_pad = np.zeros(2 * n - 1)
    h_pad[:n] = h
    
    # Use next fast FFT size
    fft_size = 1
    while fft_size < 2 * (2 * n - 1):
        fft_size *= 2
    
    f_fft = np.fft.fft(h_pad, n=fft_size)
    conv = np.fft.ifft(f_fft * np.conj(f_fft)).real[:2 * n - 1]
    
    return conv


def _compute_c3_bound(heights):
    """
    Compute the upper bound on C3 from the autoconvolution of the signed step function.
    """
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    dx = 0.5 / n  # total length 0.5, n intervals
    
    conv = _compute_convolution(h)
    max_abs = np.max(np.abs(conv))
    c3_bound = dx * max_abs
    
    return c3_bound


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1 values."""
    seq = [1]
    while len(seq) < n:
        seq += [-x for x in seq]
    return np.array(seq[:n], dtype=np.float64)


def _zadoff_chu(n, p):
    """
    Generate a Zadoff-Chu-like sequence of length n.
    h[i] = sign(sin(2*pi * i^2 / p))
    This creates a chirp sequence with multiplicative spectral properties.
    """
    i = np.arange(n)
    phase = 2 * np.pi * (i ** 2) / p
    h = np.sign(np.sin(phase))
    # Ensure no zeros (replace 0 with 1)
    h[h == 0] = 1.0
    return h


def _product_chirp(n, p1, p2):
    """
    Product of two independent Zadoff-Chu chirps with distinct prime moduli.
    This creates a sequence with specific multiplicative spectral cancellation.
    """
    h1 = _zadoff_chu(n, p1)
    h2 = _zadoff_chu(n, p2)
    h = h1 * h2
    return h


def _fast_local_search(heights, max_iter=10000, time_budget=60, start_time=0):
    """
    Fast local search using incremental convolution updates.
    When flipping bit i, the convolution changes by -2*h[i] * (h convolved with delta_i).
    This allows O(N) update instead of O(N log N) FFT per evaluation.
    """
    h = np.array(heights, dtype=np.float64)
    n = len(h)
    conv_len = 2 * n - 1
    
    # Precompute convolution
    conv = _compute_convolution(h)
    current_max = np.max(np.abs(conv))
    current_bound = (0.5 / n) * current_max
    
    if start_time == 0:
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
        best_new_max = current_max
        
        # Precompute the "influence" of each position on the convolution
        # Flipping position i changes conv by -2*h[i] * h shifted
        # We can compute this incrementally
        
        for i in range(n):
            # Compute the effect of flipping position i
            # New conv = conv - 2*h[i] * (h convolved with delta at i)
            # h convolved with delta_i is just h shifted by i positions
            # For linear convolution of length 2n-1:
            # (h * delta_i)[k] = h[k-i] if 0 <= k-i < n, else 0
            
            # The change in conv when flipping h[i]:
            # delta_conv[k] = -2 * h[i] * h[k-i] for valid k-i
            # We need to find the new max |conv + delta_conv|
            
            # Efficient approach: compute the affected range
            # h[k-i] is nonzero for k in [i, i+n-1]
            start_k = i
            end_k = min(i + n, conv_len)
            
            # Extract the affected portion
            affected = conv[start_k:end_k].copy()
            # The shift of h by i: h[k-i] for k in [i, i+n-1]
            # h[k-i] = h[0], h[1], ..., h[n-1]
            delta = -2.0 * h[i] * h[:end_k - start_k]
            
            new_affected = affected + delta
            
            # Find new max
            # The unaffected parts remain the same
            if start_k > 0:
                max_left = np.max(np.abs(conv[:start_k]))
            else:
                max_left = 0.0
            if end_k < conv_len:
                max_right = np.max(np.abs(conv[end_k:]))
            else:
                max_right = 0.0
            
            max_affected = np.max(np.abs(new_affected)) if len(new_affected) > 0 else 0.0
            new_max = max(max_left, max_affected, max_right)
            
            gain = current_max - new_max
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                best_new_max = new_max
                best_delta = delta
                best_start = start_k
                best_end = end_k
        
        if best_idx >= 0:
            # Apply the flip
            h[best_idx] = -h[best_idx]
            conv[best_start:best_end] += best_delta
            current_max = best_new_max
            current_bound = (0.5 / n) * current_max
            improved = True
    
    return h, current_bound


def _pairwise_swap_search(heights, time_budget=10, start_time=0):
    """
    Pairwise swap local search: try swapping signs at two positions to reduce C3 bound.
    Uses incremental updates for speed.
    """
    h = np.array(heights, dtype=np.float64)
    n = len(h)
    conv_len = 2 * n - 1
    
    conv = _compute_convolution(h)
    current_max = np.max(np.abs(conv))
    current_bound = (0.5 / n) * current_max
    
    if start_time == 0:
        start_time = time.time()
    
    improved = True
    while improved:
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        best_gain = 0
        best_i, best_j = -1, -1
        best_new_max = current_max
        best_delta = None
        best_ranges = None
        
        # Sample a subset of pairs to keep computation manageable
        rng = np.random.RandomState(42)
        indices = rng.choice(N, size=min(50, N), replace=False)
        
        for ii in range(len(indices)):
            i = indices[ii]
            for jj in range(ii + 1, len(indices)):
                j = indices[jj]
                if time.time() - start_time > time_budget:
                    break
                
                # Flipping both i and j
                # delta_conv = -2*h[i]*h_shifted_i - 2*h[j]*h_shifted_j
                # But after flipping i, h[i] changes, so we need to be careful
                # Actually, flipping both means:
                # new_h[i] = -h[i], new_h[j] = -h[j]
                # delta_conv[k] = -2*h[i]*h[k-i] - 2*h[j]*h[k-j]
                # (using original h values)
                
                start_k1 = i
                end_k1 = min(i + n, conv_len)
                start_k2 = j
                end_k2 = min(j + n, conv_len)
                
                start_k = min(start_k1, start_k2)
                end_k = max(end_k1, end_k2)
                
                affected = conv[start_k:end_k].copy()
                
                # Add contribution from flipping i
                for k in range(start_k, end_k):
                    ki = k - i
                    if 0 <= ki < n:
                        affected[k - start_k] += -2.0 * h[i] * h[ki]
                    kj = k - j
                    if 0 <= kj < n:
                        affected[k - start_k] += -2.0 * h[j] * h[kj]
                
                if start_k > 0:
                    max_left = np.max(np.abs(conv[:start_k]))
                else:
                    max_left = 0.0
                if end_k < conv_len:
                    max_right = np.max(np.abs(conv[end_k:]))
                else:
                    max_right = 0.0
                
                max_affected = np.max(np.abs(affected)) if len(affected) > 0 else 0.0
                new_max = max(max_left, max_affected, max_right)
                
                gain = current_max - new_max
                if gain > best_gain:
                    best_gain = gain
                    best_i, best_j = i, j
                    best_new_max = new_max
                    best_delta = affected - conv[start_k:end_k]
                    best_ranges = (start_k, end_k)
        
        if best_i >= 0:
            h[best_i] = -h[best_i]
            h[best_j] = -h[best_j]
            conv[best_ranges[0]:best_ranges[1]] += best_delta
            current_max = best_new_max
            current_bound = (0.5 / n) * current_max
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
    
    # Try product chirp sequences with distinct prime moduli
    prime_pairs = [(401, 409), (401, 419), (409, 419), (401, 421), (409, 421), (419, 421)]
    for p1, p2 in prime_pairs:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        h_chirp = _product_chirp(N, p1, p2)
        bound = _compute_c3_bound(h_chirp)
        if bound < best_bound:
            best_bound = bound
            best_h = h_chirp.copy()
    
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
    
    # Fast local search from best initialization
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 15:
        local_time = min(remaining - 10, 60)
        best_h, best_bound = _fast_local_search(best_h, max_iter=10000, time_budget=local_time, start_time=start_time)
    
    # Additional restarts if time permits
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 20:
        for trial in range(5):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            # Mix of random and chirp-based restarts
            if trial % 2 == 0:
                h = np.random.choice([-1.0, 1.0], size=N)
            else:
                p1, p2 = prime_pairs[trial % len(prime_pairs)]
                h = _product_chirp(N, p1, p2)
            
            bound = _compute_c3_bound(h)
            
            if bound < best_bound:
                local_time = min(TIME_LIMIT - (time.time() - start_time) - 5, 20)
                h, bound = _fast_local_search(h, max_iter=5000, time_budget=local_time, start_time=start_time)
                if bound < best_bound:
                    best_bound = bound
                    best_h = h.copy()
    
    # Pairwise swap search for further refinement
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 10:
        swap_time = min(remaining - 5, 20)
        best_h, best_bound = _pairwise_swap_search(best_h, time_budget=swap_time, start_time=start_time)
    
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

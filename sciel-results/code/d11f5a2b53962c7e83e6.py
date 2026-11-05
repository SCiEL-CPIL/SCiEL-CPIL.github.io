"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

TIME_LIMIT = 145  # Safety margin before 150s kill
DX = 0.5 / N

def _compute_sidelobe_bound(heights):
    """
    Compute the upper bound on C3 based on the maximum absolute value
    of the non-zero lags of the autoconvolution.
    """
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    
    # Linear autocorrelation via FFT
    # Pad to 2*n to avoid circular convolution artifacts for lags up to n-1
    # Actually, for linear autocorrelation of length n, we need size >= 2n-1.
    # Using FFT size 2*n is safe and fast (power of 2).
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
        
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    
    # The autocorrelation is symmetric. acf[k] is the sum of h[i]*h[i+k].
    # Lag 0 is acf[0] = sum h[i]^2 = N (for +/-1).
    # We want max abs of lags 1 to n-1.
    # Note: acf from irfft gives lags 0 to fft_size/2.
    # We need lags 1 to n-1.
    max_sidelobe = np.max(np.abs(acf[1:n]))
    
    return DX * max_sidelobe

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1 values."""
    seq = [1]
    while len(seq) < n:
        seq += [-x for x in seq]
    return np.array(seq[:n], dtype=np.float64)

def _local_search_fast(h, time_budget, start_time=0):
    """
    Fast local search using O(N) analytic updates for the convolution.
    Flips a single bit at a time if it reduces the max sidelobe.
    """
    h = np.array(h, dtype=np.float64)
    n = len(h)
    
    if start_time == 0:
        start_time = time.time()
        
    # Initial computation
    current_bound = _compute_sidelobe_bound(h)
    
    # Precompute FFT for initial conv? No, we need the actual conv array for updates.
    # Let's compute the linear autocorrelation array explicitly.
    # Using FFT for initial setup is fast.
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    conv = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # We only care about conv[0:n] effectively (lags 0 to n-1)
    # conv[k] for k in 0..n-1 is the autocorrelation at lag k.
    
    improved = True
    while improved and time.time() - start_time < time_budget:
        improved = False
        best_gain = 0.0
        best_idx = -1
        
        # Try flipping each position
        for i in range(n):
            old_val = h[i]
            # Perturbation:
            # conv_new[k] = conv[k] - 2 * old_val * (h[i+k] + h[i-k])
            # where h[idx] is 0 if idx out of [0, n-1]
            
            # We need to update conv[1:n] and check the max.
            # Instead of creating a new array, we can compute the new max efficiently?
            # Or just update a copy of the relevant part.
            
            # For speed, we compute the perturbation vector and add it to conv.
            # But doing this for every i is O(N^2) per pass.
            # Let's try to vectorize the perturbation calculation.
            
            # Perturbation for flip i:
            # delta_conv[k] = -2 * h[i] * (h[i+k] + h[i-k])
            
            # We can precompute the shifts?
            # For a given i, the perturbation affects all k.
            
            # Let's just do the O(N) update on a temporary array for the candidate.
            # Since N=400, this is fast.
            
            # Create a view of conv we care about (lags 1 to n-1)
            # We need to check if max(abs(conv[1:n] + delta[1:n])) is smaller.
            
            # Compute delta for all k in 1..n-1
            # h[i+k] is valid if 0 <= i+k < n
            # h[i-k] is valid if 0 <= i-k < n
            
            # Vectorized approach for a single i:
            # This is still a loop over i.
            
            # Let's optimize the inner check.
            # We can precompute the "neighborhood" of h.
            
            # For now, simple loop.
            # To speed up, we can break early if the potential gain is small?
            
            # Calculate delta_conv for k in 1..n-1
            # k range: 1 to n-1
            # term1: h[i+k]
            # term2: h[i-k]
            
            # Let's construct the delta array
            delta = np.zeros(n) # We only use 1..n-1
            
            # Indices for h[i+k]
            # k in 1..n-1
            # i+k in i+1 .. i+n-1
            # Valid if i+k < n -> k < n-i
            # So for k in 1..min(n-1, n-i-1), h[i+k] is h[i+k]
            
            # This is getting complex to vectorize per i without overhead.
            # Given N=400, a simple loop in Python might be too slow for 150s if we do many passes.
            # Let's estimate:
            # 400 flips to check.
            # Each check: O(N) operations in Python? No, use NumPy.
            # delta = -2 * h[i] * (np.roll(h, -i) + np.roll(h, i))
            # This is O(N) in NumPy (C speed).
            # 400 * O(N) = 400 * 400 = 160k ops.
            # NumPy overhead per call is ~1-5us.
            # 400 calls * 5us = 2ms per pass.
            # 150s / 2ms = 75,000 passes.
            # This is plenty.
            
            # Let's use np.roll for the perturbation.
            # Note: np.roll wraps around. We need linear correlation.
            # h is length N.
            # h[i+k] for linear conv:
            # If we use np.roll(h, -i), element k becomes h[k-i mod N].
            # This is circular.
            # We need linear.
            
            # Let's stick to the explicit update with slicing, which is also fast.
            
            # Actually, let's just recompute the bound for the candidate?
            # No, that's O(N log N).
            # The O(N) update is better.
            
            # Let's implement the O(N) update correctly.
            # delta[k] = -2 * h[i] * (h[i+k] + h[i-k])
            
            # We can precompute the "shifted" arrays? No, i changes.
            
            # Let's try a different approach:
            # Just recompute the FFT for the candidate?
            # FFT of size 512 (next power of 2 for 2*400) is very fast.
            # ~10-20us.
            # 400 candidates * 20us = 8ms per pass.
            # 150s / 8ms = 18,000 passes.
            # This is also fine and much simpler to implement correctly.
            
            # Let's use the FFT recomputation for simplicity and robustness.
            
            h[i] = -old_val
            new_bound = _compute_sidelobe_bound(h)
            h[i] = old_val
            
            gain = current_bound - new_bound
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_idx >= 0:
            h[best_idx] = -h[best_idx]
            current_bound -= best_gain
            improved = True
        else:
            # Stuck in local minimum.
            # Try a random kick (flip 2-3 bits) to escape?
            # Or break.
            # Let's try a random kick to escape local minima.
            if time.time() - start_time < time_budget - 1:
                # Flip 2 random bits
                idxs = np.random.choice(n, size=2, replace=False)
                h[idxs] = -h[idxs]
                current_bound = _compute_sidelobe_bound(h)
                # Don't mark as improved, just continue searching from this new state
                # But we need to be careful not to loop forever.
                # Let's just break for now, or do a limited number of kicks.
                # For now, break to avoid wasting time if stuck.
                break

    return h, current_bound

def construct():
    start_time = time.time()
    
    # Initialize with multiple starting points
    best_h = np.ones(N)
    best_bound = _compute_sidelobe_bound(best_h)
    
    # Try Thue-Morse sequence
    tm = _thue_morse(N)
    tm_bound = _compute_sidelobe_bound(tm)
    if tm_bound < best_bound:
        best_bound = tm_bound
        best_h = tm.copy()
    
    # Try random initializations
    np.random.seed(42)
    for trial in range(10):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        h = np.random.choice([-1.0, 1.0], size=N)
        bound = _compute_sidelobe_bound(h)
        if bound < best_bound:
            best_bound = bound
            best_h = h.copy()
    
    # Local search from best initialization
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 15:
        local_time = min(remaining - 10, 60)
        best_h, best_bound = _local_search_fast(best_h, local_time, start_time=start_time)
    
    # Additional restarts if time permits
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 20:
        for trial in range(5):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            h = np.random.choice([-1.0, 1.0], size=N)
            bound = _compute_sidelobe_bound(h)
            
            if bound < best_bound:
                local_time = min(TIME_LIMIT - (time.time() - start_time) - 5, 20)
                h, bound = _local_search_fast(h, local_time, start_time=start_time)
                if bound < best_bound:
                    best_bound = bound
                    best_h = h.copy()
    
    # Final verification
    final_bound = _compute_sidelobe_bound(best_h)
    
    return best_h

# Test
if __name__ == "__main__":
    h = construct()
    print(f"Length: {len(h)}")
    print(f"First 20: {h[:20]}")
    print(f"Sidelobe C3 bound: {_compute_sidelobe_bound(h)}")
    # Assuming seed value is approx 0.49875 (all ones)
    seed_val = 0.49875
    print(f"Score: {seed_val / _compute_sidelobe_bound(h)}")
# EVOLVE-BLOCK-END

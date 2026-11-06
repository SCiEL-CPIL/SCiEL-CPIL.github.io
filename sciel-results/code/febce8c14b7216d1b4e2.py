import time
import numpy as np
from scipy.signal import fftconvolve
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin for 120s limit
    
    # Generate powerful numbers up to L
    # A powerful number has the form a^2 * b^3 where gcd(a,b)=1
    # But it's simpler to just generate all a^2 * b^3 <= L and deduplicate
    # The count of powerful numbers up to X is ~ sqrt(X) + X^(1/3) - ...
    # For L = 10^9, we get roughly 30k-40k powerful numbers
    
    L = 10**9
    max_b = int(L ** (1.0/3.0)) + 1  # ~1000
    
    powerfuls_list = []
    for b in range(1, max_b + 1):
        b3 = b ** 3
        max_a = isqrt(L // b3)
        if max_a < 1:
            break
        # Vectorized: a^2 * b3 for a = 1..max_a
        a_vals = np.arange(1, max_a + 1, dtype=np.int64)
        vals = a_vals * a_vals * b3
        vals = vals[vals <= L]
        if len(vals) > 0:
            powerfuls_list.append(vals)
    
    if not powerfuls_list:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    powerfuls_arr = np.concatenate(powerfuls_list)
    powerfuls_arr = np.unique(powerfuls_arr)
    powerfuls_arr = powerfuls_arr[powerfuls_arr <= L]
    
    elapsed_gen = time.time() - start_time
    
    # Check time
    if time.time() > deadline - 60:
        # Not enough time for FFT, use sparse approach
        return _sparse_search(powerfuls_arr, deadline, start_time)
    
    # For FFT, we need a dense array of size 2*L+1
    # This is feasible for L up to about 2*10^8 (array of size 4*10^8 int64 = 3.2GB)
    # But memory is a concern. Let's use a smaller L for the dense FFT approach
    # or use the sparse approach with bincount
    
    # Strategy: Use np.bincount on all pairwise sums
    # M = len(powerfuls_arr), pairs = M*(M+1)/2
    # For M=30000, pairs ~ 450M which is too many for memory
    # 
    # Alternative: Use FFT with a moderate L
    # Let's try L_fft = 2*10^8 which gives array size 4*10^8
    # That's 3.2 GB for int64, too much. Use float32 or int32.
    # Actually fftconvolve handles this internally.
    
    # Better strategy: Use the sparse approach with bincount on chunks
    # Or: reduce L for FFT to something manageable
    
    # Let's use FFT with L_fft = 10^8 (array size 2*10^8, ~1.6GB as float64)
    # This should be feasible and fast
    L_fft = min(L, 120_000_000)
    pow_for_fft = powerfuls_arr[powerfuls_arr <= L_fft]
    
    if len(pow_for_fft) < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    size = L_fft + 1
    arr = np.zeros(size, dtype=np.float64)
    arr[pow_for_fft] = 1.0
    
    # Compute convolution
    counts = fftconvolve(arr, arr, mode='full')
    
    # Find best N in range [2, 2*L_fft]
    search_slice = counts[2:2*L_fft+1]
    if len(search_slice) == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Structural Change: Instead of just argmax or random sampling,
    # get the top K candidates from the FFT result.
    # The FFT result is approximate, so we verify the top few.
    
    # Get indices of top 100 values in the search slice
    # Note: argpartition is faster than argsort for top-K
    k_candidates = min(100, len(search_slice))
    top_k_indices = np.argpartition(search_slice, -k_candidates)[-k_candidates:]
    
    # Sort these top K indices by their FFT count (descending)
    top_k_indices = top_k_indices[np.argsort(search_slice[top_k_indices])[::-1]]
    
    powerfuls_set = set(powerfuls_arr.tolist())
    powerfuls_sorted = powerfuls_arr.tolist()
    
    best_N = -1
    best_pairs = []
    best_count = 0
    
    # Verify the top K candidates
    for idx in top_k_indices:
        if time.time() > deadline - 15:
            break
            
        N_cand = int(idx + 2)
        
        # Count representations for N_cand
        # Iterate through powerfuls a <= N_cand // 2
        # Check if N_cand - a is in powerfuls_set
        count = 0
        pairs = []
        max_a = N_cand // 2
        
        # Optimization: Only iterate up to max_a
        # powerfuls_sorted is sorted. We can break early.
        for a in powerfuls_sorted:
            if a > max_a:
                break
            b = N_cand - a
            if b in powerfuls_set:
                count += 1
                pairs.append([a, b])
        
        if count > best_count:
            best_count = count
            best_N = N_cand
            best_pairs = pairs
    
    if best_N == -1:
        # Fallback to argmax if no candidates verified (should not happen)
        best_N = int(np.argmax(search_slice) + 2)
        best_pairs = []
        max_a = best_N // 2
        for a in powerfuls_sorted:
            if a > max_a:
                break
            b = best_N - a
            if b in powerfuls_set:
                best_pairs.append([a, b])
        best_pairs.sort()
    else:
        best_pairs.sort()

    return {'N': best_N, 'pairs': best_pairs}

def _sparse_search(powerfuls_arr, deadline, start_time):
    """Fallback sparse search when FFT is too expensive."""
    powerfuls_set = set(powerfuls_arr.tolist())
    powerfuls_sorted = powerfuls_arr.tolist()
    n_pow = len(powerfuls_sorted)
    
    if n_pow < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_N = 2
    best_count = 0
    
    # Use bincount on chunked pairwise sums
    # Process in chunks to avoid memory issues
    chunk_size = min(n_pow, 5000)
    
    for start in range(0, n_pow, chunk_size):
        if time.time() > deadline - 20:
            break
        end = min(start + chunk_size, n_pow)
        chunk = powerfuls_sorted[start:end]
        
        # Compute all sums within this chunk and with the full set
        # This is O(chunk_size * n_pow) which can be large
        # Instead, just compute sums within the chunk
        sums = np.array([a + b for a in chunk for b in chunk if a <= b], dtype=np.int64)
        if len(sums) > 0:
            counts = np.bincount(sums)
            max_count = int(counts.max())
            if max_count > best_count:
                best_count = max_count
                best_N = int(counts.argmax())
    
    # Verify and get pairs for best_N
    pairs = []
    max_a = best_N // 2
    for a in powerfuls_sorted:
        if a > max_a:
            break
        b = best_N - a
        if b in powerfuls_set:
            pairs.append([a, b])
    
    pairs.sort()
    return {'N': best_N, 'pairs': pairs}

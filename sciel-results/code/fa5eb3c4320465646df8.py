import time
import numpy as np
from math import isqrt

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    powerful = set()
    max_b = int(limit ** (1.0 / 3.0)) + 2
    
    # Sieve for square-free numbers up to max_b
    is_sq_free = [True] * (max_b + 1)
    if max_b >= 0:
        is_sq_free[0] = False
    if max_b >= 1:
        is_sq_free[1] = True # 1 is square-free
        
    for i in range(2, isqrt(max_b) + 1):
        sq = i * i
        for j in range(sq, max_b + 1, sq):
            is_sq_free[j] = False
            
    for b in range(1, max_b + 1):
        if not is_sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            continue
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            powerful.add(a * a * b3)
            
    return powerful

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin for 120s limit
    
    # Try a larger limit first, fallback if too slow/memory error
    limits_to_try = [10**7, 5*10**6, 2*10**6]
    
    best_result = None
    
    for L in limits_to_try:
        if time.time() > deadline - 10:
            break
            
        try:
            # Generate powerful numbers
            powerful = _generate_powerful(L)
            if not powerful:
                continue
                
            # Create array for convolution
            size = 2 * L + 1
            # Use float64 for FFT
            arr = np.zeros(size, dtype=np.float64)
            for p in powerful:
                arr[p] = 1.0
                
            # Determine FFT size (power of 2)
            n_fft = 1
            while n_fft < 2 * size:
                n_fft *= 2
                
            # Compute convolution
            f = np.fft.fft(arr, n=n_fft)
            f2 = f * f
            conv = np.fft.ifft(f2).real
            
            # Find best N
            best_N = 2
            max_count = 0
            
            # Iterate through possible N
            # We can optimize by only checking up to 2*L
            # conv is length n_fft, but we only care up to 2*L
            # Note: conv[i] corresponds to sum i
            for N in range(2, 2 * L + 1):
                c = int(round(conv[N]))
                if c > max_count:
                    max_count = c
                    best_N = N
                    
            # Extract pairs
            pairs = []
            p_list = sorted(powerful)
            p_set = powerful
            # Binary search or simple iteration
            # Since p_list is sorted, we can break early
            for a in p_list:
                if 2 * a > best_N:
                    break
                b = best_N - a
                if b in p_set:
                    pairs.append([a, b])
                    
            best_result = {'N': best_N, 'pairs': pairs}
            
            # If we found a good result, we can stop or try larger?
            # The loop tries larger L first. So if we succeed with 10^7, we are done.
            # If it crashes, we catch and try next.
            break
            
        except MemoryError:
            continue
        except Exception:
            continue

    if best_result is None:
        # Fallback to simple search if all failed
        L = 10**5
        powerful = _generate_powerful(L)
        p_set = powerful
        best_N = 2
        max_pairs = []
        for N in range(2, 2*L+1):
            pairs = []
            for a in p_set:
                if 2*a > N: break
                if N-a in p_set:
                    pairs.append([a, N-a])
            if len(pairs) > len(max_pairs):
                max_pairs = pairs
                best_N = N
        best_result = {'N': best_N, 'pairs': max_pairs}
        
    return best_result

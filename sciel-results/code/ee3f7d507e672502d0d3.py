import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Strategy: Maximize M to increase the density of powerful numbers in the search space.
    # Use a fallback mechanism if memory limits are exceeded.
    
    # Try a large M first
    target_Ms = [40_000_000, 30_000_000, 20_000_000, 10_000_000]
    
    powerful_list = []
    powerful_set = set()
    M_used = 0
    
    for M in target_Ms:
        try:
            if time.time() > deadline - 20:
                break
                
            # 1. Generate powerful numbers up to M
            powerful_set = set()
            max_v = int(M ** (1/3)) + 2
            
            for v in range(1, max_v + 1):
                v_cubed = v ** 3
                if v_cubed > M:
                    break
                max_u = isqrt(M // v_cubed)
                for u in range(1, max_u + 1):
                    p = u * u * v_cubed
                    powerful_set.add(p)
                    
            powerful_list = sorted(powerful_set)
            K = len(powerful_list)
            M_used = M
            
            # 2. Prepare for FFT
            size = 1
            while size < 2 * M + 1:
                size *= 2
                
            # Check if size is feasible (heuristic: < 2^27 for 1GB memory with float64)
            if size > 2**27:
                continue # Try smaller M
                
            # Create indicator array
            A = np.zeros(size, dtype=np.float64)
            for p in powerful_list:
                A[p] = 1.0
                
            # 3. FFT Convolution
            F = np.fft.rfft(A)
            F_sq = F * F
            counts_float = np.fft.irfft(F_sq, n=size)
            
            # We only care about N up to 2*M
            counts_int = np.round(counts_float[:2*M + 1]).astype(np.int64)
            
            # 4. Find candidates
            max_val = np.max(counts_int)
            if max_val == 0:
                continue
                
            top_k = 500
            top_indices = np.argpartition(counts_int, -top_k)[-top_k:]
            
            # 5. Verify exact counts for these candidates
            best_N = 0
            best_exact_count = 0
            best_pairs = []
            
            powerful_set_local = powerful_set
            powerful_list_local = powerful_list
            
            for idx in top_indices:
                N = int(idx)
                count = 0
                pairs = []
                # Iterate through powerful numbers <= N/2
                # Use binary search to limit the loop? 
                # powerful_list is sorted.
                
                # Find index of first element > N//2
                # Since N <= 2*M, and powerful_list <= M, we just iterate all K.
                # K is ~15000 for M=40M. 500 * 15000 = 7.5M checks. Fast.
                
                for a in powerful_list_local:
                    if a > N // 2:
                        break
                    b = N - a
                    if b in powerful_set_local:
                        count += 1
                        pairs.append([a, b])
                        
                if count > best_exact_count:
                    best_exact_count = count
                    best_N = N
                    best_pairs = pairs
                    
            # If we found a good result, return it
            if best_exact_count > 0:
                return {'N': best_N, 'pairs': best_pairs}
                
        except MemoryError:
            continue
        except Exception:
            continue

    # Fallback
    return {'N': 2, 'pairs': [[1, 1]]}

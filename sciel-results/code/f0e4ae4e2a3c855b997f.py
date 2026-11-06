import time
import sys
import numpy as np
from math import isqrt

def _generate_powerful_numbers(limit):
    """Generate all powerful numbers up to limit using u^2 * v^3 representation."""
    powerful = set()
    max_v = int(limit ** (1.0 / 3.0)) + 2
    
    for v in range(1, max_v + 1):
        v3 = v * v * v
        if v3 > limit:
            break
        max_u = isqrt(limit // v3)
        for u in range(1, max_u + 1):
            val = u * u * v3
            if val <= limit:
                powerful.add(val)
            else:
                break
    return sorted(powerful)

def _count_pairs_for_N(N, powerful_list, powerful_set):
    """Count pairs (a, b) with a <= b, a + b = N, both powerful."""
    count = 0
    pairs = []
    half_N = N // 2
    for a in powerful_list:
        if a > half_N:
            break
        b = N - a
        if b in powerful_set:
            count += 1
            pairs.append([a, b])
    return count, pairs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_result = {'N': 5, 'pairs': [[1, 4]]}
    
    # Try multiple strategies with increasing M
    # Strategy 1: FFT for moderate M
    limits_to_try = [80_000_000, 50_000_000, 30_000_000]
    
    for M in limits_to_try:
        try:
            if time.time() > deadline - 40:
                break
                
            # 1. Generate powerful numbers
            gen_start = time.time()
            powerful_list = _generate_powerful_numbers(M)
            gen_time = time.time() - gen_start
            
            if time.time() > deadline - 25:
                continue
                
            K = len(powerful_list)
            if K == 0:
                continue
            
            # 2. FFT Convolution
            S = 2 * M + 1
            
            # Use float32 to save memory
            arr = np.zeros(S, dtype=np.float32)
            arr[powerful_list] = 1.0
            
            # Perform FFT
            fft_arr = np.fft.rfft(arr)
            conv = np.fft.irfft(fft_arr * fft_arr, n=S)
            
            # 3. Find best N
            top_k = 200
            if S > top_k:
                top_indices = np.argpartition(conv, -top_k)[-top_k:]
            else:
                top_indices = np.arange(S)
                
            top_indices = top_indices[np.argsort(conv[top_indices])[::-1]]
            
            best_N = 2
            best_count = 1
            best_pairs = [[1, 1]]
            
            powerful_set = set(powerful_list)
            
            # Verify top candidates
            for idx in top_indices:
                if time.time() > deadline - 5:
                    break
                    
                N_cand = int(idx)
                
                count, exact_pairs = _count_pairs_for_N(N_cand, powerful_list, powerful_set)
                
                if count > best_count:
                    best_count = count
                    best_N = N_cand
                    best_pairs = exact_pairs
                    
                # Early termination
                if conv[idx] < 2 * best_count:
                    break
            
            # Update best result
            if best_count > len(best_result['pairs']):
                best_result = {'N': best_N, 'pairs': best_pairs}
                
            return best_result

        except MemoryError:
            continue
        except Exception:
            continue

    return best_result

if __name__ == "__main__":
    res = construct()
    print(res['N'], len(res['pairs']))

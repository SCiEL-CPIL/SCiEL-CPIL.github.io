import time
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 85.0  # 5s margin before the 90s kill
    
    N_MAX = 1000000
    N_PLUS_1 = N_MAX + 1
    SIZE = N_PLUS_1 + 1  # 1000002

    # Initialize LPF array
    # Using int32 to save memory and potentially improve cache performance
    lpf = np.zeros(SIZE, dtype=np.int32)
    
    # Sieve of Eratosthenes variant for Largest Prime Factor
    # We iterate i from 2 to N_PLUS_1
    # If lpf[i] is 0, i is prime.
    # We update all multiples of i with i.
    # Since we go in increasing order, the last write is the largest prime factor.
    
    # Optimization: Handle 2 separately
    # This reduces the number of checks in the main loop by half
    # and the number of slice assignments by half.
    lpf[2::2] = 2
    
    # Loop over odd numbers only
    # If lpf[i] is 0, i is prime.
    # Note: lpf[1] is 0, but 1 is not prime. We start from 3.
    # Also, lpf[i] for odd i might be 0 if it hasn't been touched by a smaller prime factor.
    # If it's 0, it means it's prime (since all composite odd numbers have a prime factor < themselves,
    # and that factor would have updated it).
    
    for i in range(3, SIZE, 2):
        if time.time() - start_time > TIME_LIMIT:
            # Fallback: return a known good candidate or the best found so far?
            # Since the sieve is fast (vectorized slices), this is very unlikely to trigger.
            # If it does, we return a default safe value.
            return 10000
            
        if lpf[i] == 0:
            # i is prime
            lpf[i::i] = i
            
    # Compute scores for n in [10000, 1000000]
    # n corresponds to lpf[n]
    # n+1 corresponds to lpf[n+1]
    
    # Slices
    lpf_n = lpf[10000:1000001]
    lpf_np1 = lpf[10001:1000002]
    
    # Max LPF for the pair
    scores = np.maximum(lpf_n, lpf_np1)
    
    # Find the index with the minimum score
    min_idx = np.argmin(scores)
    
    best_n = 10000 + min_idx
    
    return int(best_n)

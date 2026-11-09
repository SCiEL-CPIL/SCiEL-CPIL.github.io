import numpy as np

def construct():
    # Range is [10000, 1000000]
    # We need largest prime factor up to 1000001
    N = 1000001
    # Initialize array to 1
    # L[i] will store the largest prime factor of i
    L = np.ones(N + 1, dtype=np.int64)
    
    # Sieve to find largest prime factors
    # Iterate through potential primes
    for i in range(2, N + 1):
        if L[i] == 1:
            # i is prime
            # Set largest prime factor for all multiples of i
            # Since we iterate i in increasing order, the last assignment
            # for any composite number j will be its largest prime factor.
            L[i::i] = i
            
    # We want to minimize max(L[n], L[n+1]) for n in [10000, 1000000]
    # Create arrays for the range
    # n ranges from 10000 to 1000000
    # So L[n] ranges from L[10000] to L[1000000]
    # L[n+1] ranges from L[10001] to L[1000001]
    
    start = 10000
    end = 1000000
    
    # Slice L for n and n+1
    # L[start : end+1] corresponds to L[n]
    # L[start+1 : end+2] corresponds to L[n+1]
    
    vals_n = L[start : end + 1]
    vals_np1 = L[start + 1 : end + 2]
    
    # Compute max for each pair
    max_vals = np.maximum(vals_n, vals_np1)
    
    # Find index of minimum value
    min_idx = np.argmin(max_vals)
    
    # The corresponding n
    best_n = start + min_idx
    
    return best_n

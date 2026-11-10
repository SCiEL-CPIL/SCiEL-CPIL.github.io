import numpy as np
from collections import Counter

def construct():
    max_n = 1000000
    window_size = 24
    limit = max_n + window_size + 10
    
    # 1. Compute Smallest Prime Factor (SPF)
    # Using a standard sieve approach which is fast for N=1e6
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i: # i is prime
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    # 2. Compute g[m] = product of prime factors < 24
    g = [1] * (limit + 1)
    for m in range(2, limit + 1):
        p = spf[m]
        if p < 24:
            g[m] = g[m // p] * p
        else:
            g[m] = g[m // p]
            
    # 3. Sliding Window
    # We look for n in [1000, 1000000]
    # Window is [n+1, ..., n+24]
    # Indices in g: n+1 to n+24
    
    # Use an array for counts to avoid dictionary overhead
    # Max value of g[m] is m, which is <= limit
    counts = [0] * (limit + 1)
    
    # Initialize window for n=1000
    start_idx = 1001
    end_idx = 1001 + window_size # exclusive
    
    current_distinct = 0
    
    # Initialize counts
    for i in range(start_idx, end_idx):
        val = g[i]
        if counts[val] == 0:
            current_distinct += 1
        counts[val] += 1
            
    min_distinct = current_distinct
    best_n = 1000
    
    # Slide
    # Next n is n+1
    # Window becomes [n+2, ..., n+25]
    # Remove g[n+1], Add g[n+25]
    
    for n in range(1000, max_n):
        # Remove index n+1
        out_val = g[n+1]
        counts[out_val] -= 1
        if counts[out_val] == 0:
            current_distinct -= 1
            
        # Add index n+25
        in_val = g[n+25]
        if counts[in_val] == 0:
            current_distinct += 1
        counts[in_val] += 1
            
        if current_distinct < min_distinct:
            min_distinct = current_distinct
            best_n = n + 1
            
    return best_n

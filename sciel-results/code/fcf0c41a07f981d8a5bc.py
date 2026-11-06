import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Determine limit L
    # Start with a conservative limit, increase if time allows?
    # FFT size is the bottleneck.
    # Let's try L = 4 * 10^7
    L = 40_000_000
    
    # 1. Generate powerful numbers up to L
    powerful_set = set()
    # b^3 * a^2
    max_b = int(L ** (1/3)) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = int(math.isqrt(L // b3))
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerful_set.add(n)
    
    powerful_list = sorted(powerful_set)
    powerful_bool = np.zeros(L + 1, dtype=np.uint8)
    for p in powerful_list:
        powerful_bool[p] = 1
    
    # 2. FFT Convolution
    # Next power of 2 >= 2*L
    size = 1
    while size < 2 * L:
        size *= 2
        
    # Check memory/time feasibility roughly
    # If size is too big, reduce L? 
    # For L=4e7, size=1.34e8. 
    # This is large. Let's try to be safe with L=2e7 first?
    # Let's stick to L=20_000_000 to ensure it runs within time/memory.
    # Re-evaluate: 2e7 -> size 2^26 = 6.7e7. Memory ~1GB. Safe.
    
    L = 20_000_000
    powerful_set = set()
    max_b = int(L ** (1/3)) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = int(math.isqrt(L // b3))
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerful_set.add(n)
    
    powerful_list = sorted(powerful_set)
    powerful_bool = np.zeros(L + 1, dtype=np.uint8)
    for p in powerful_list:
        powerful_bool[p] = 1
        
    size = 1
    while size < 2 * L:
        size *= 2
        
    # Pad to size
    f = np.fft.fft(powerful_bool, n=size)
    f_sq = f * f
    conv = np.fft.ifft(f_sq)
    
    # Real part, round to nearest int
    counts = np.round(conv.real).astype(np.int64)
    
    # counts[k] is the number of ordered pairs (x, y) with x+y=k
    # We want distinct pairs a <= b.
    # count_distinct = (counts[k] + is_powerful(k//2) if k%2==0 else 0) // 2
    
    # Find best K
    best_K = 0
    best_count = 0
    
    # Optimization: only check K where counts[k] > 0
    # But we need to calculate the exact distinct count.
    # We can precompute a set of powerful numbers for O(1) lookup.
    # We already have powerful_set.
    
    # Iterate through possible K
    # To speed up, we can find indices where counts > 0
    non_zero_indices = np.nonzero(counts[:L+1])[0]
    
    for k in non_zero_indices:
        if k == 0:
            continue
        c = counts[k]
        if c == 0:
            continue
        
        # Calculate distinct pairs
        # If k is even and k/2 is powerful, add 1 to c to make it divisible by 2?
        # Formula: (c + delta) / 2
        # where delta = 1 if (k%2==0 and k//2 in powerful_set) else 0
        # Actually, if x=y, it contributes 1 to c. The pair is {x,x}. Count = 1.
        # If x!=y, they contribute 2 to c. The pair is {x,y}. Count = 1.
        # So count = (c - 1) // 2 + 1 if self_pair else c // 2
        # Simpler: 
        # Let S be the set of powerful numbers.
        # Pairs are {x, k-x} with x < k-x.
        # c = sum_{x in S, x<k} 1_{k-x in S}
        # If x = k-x (k even, x=k/2 in S), this term is 1.
        # If x != k-x, the terms x and k-x are both in S, so they contribute 2 to c.
        # So c = 2 * (# pairs with x < k-x) + (1 if self_pair else 0)
        # # pairs = (c - (1 if self_pair else 0)) // 2 + (1 if self_pair else 0)
        # # pairs = (c + (1 if self_pair else 0)) // 2
        
        self_pair = (k % 2 == 0 and (k // 2) in powerful_set)
        distinct_count = (c + (1 if self_pair else 0)) // 2
        
        if distinct_count > best_count:
            best_count = distinct_count
            best_K = k
    
    if best_K == 0:
        # Fallback
        return {'N': 5, 'pairs': [[1, 4]]}
        
    # Extract pairs
    pairs = []
    half_K = best_K // 2
    for a in powerful_list:
        if a > half_K:
            break
        b = best_K - a
        if b in powerful_set:
            pairs.append([a, b])
            
    return {'N': best_K, 'pairs': pairs}

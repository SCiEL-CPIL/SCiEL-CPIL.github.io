import math
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit, 10s buffer
    
    # 1. Generate powerful numbers up to L
    # L = 10^11 gives ~13M numbers.
    # L = 2*10^11 gives ~22M numbers.
    # Let's try 2*10^11 if fast enough.
    L = 2 * 10**11
    
    powerful_set = set()
    max_b = int(L ** (1/3)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        max_a = math.isqrt(L // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= L:
                powerful_set.add(val)
        
        if time.time() > deadline - 30:
            break
            
    # Convert to sorted list for sampling and verification
    powerful_list = sorted(powerful_set)
    N_pow = len(powerful_list)
    
    # 2. Sample candidates
    # We want N close to 2*L to maximize potential pairs.
    # Sample from the top part of the list.
    sample_size = 2000
    # If N_pow is small, adjust
    if N_pow < sample_size * 2:
        sample_size = max(100, N_pow // 2)
        
    # Take top 20% of powerful numbers for sampling
    top_start = max(0, N_pow - int(N_pow * 0.2))
    top_part = powerful_list[top_start:]
    
    # Randomly sample from top_part
    A_sample = random.sample(top_part, min(sample_size, len(top_part)))
    B_sample = random.sample(top_part, min(sample_size, len(top_part)))
    
    # Compute sums and count
    sums_count = {}
    for a in A_sample:
        for b in B_sample:
            s = a + b
            if s <= 10**18:
                sums_count[s] = sums_count.get(s, 0) + 1
                
    # Get top 5 candidates
    if not sums_count:
        # Fallback if no sums found (should not happen)
        return {'N': 2, 'pairs': [[1, 1]]}
        
    top_candidates = sorted(sums_count.items(), key=lambda x: x[1], reverse=True)[:5]
    
    best_N = 0
    best_pairs = []
    best_score = 0
    
    # 3. Verify candidates
    for N, _ in top_candidates:
        if time.time() > deadline - 10:
            break
            
        pairs = []
        # Iterate through powerful numbers a <= N//2
        # Use binary search to find upper bound index
        # But powerful_list is sorted.
        # We can just iterate and break.
        for a in powerful_list:
            if a > N // 2:
                break
            b = N - a
            if b in powerful_set:
                pairs.append([a, b])
                
        if len(pairs) > best_score:
            best_score = len(pairs)
            best_N = N
            best_pairs = pairs
            
    if best_N == 0:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': best_pairs}

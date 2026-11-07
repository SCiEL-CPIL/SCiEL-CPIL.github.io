import numpy as np
import time
from bisect import bisect_left

def construct():
    t0 = time.time()
    hard_stop = t0 + 85.0  # Safety margin before 90s kill

    N = 10000000

    # 1. Compute sigma(n) for all n up to N using vectorized sieve
    # This is the standard efficient way to compute sigma for all n.
    # sigma[n] = sum of divisors of n.
    # Using the property: sigma is multiplicative, but sieve is faster for full array.
    # sigma[n] = sum_{d|n} d
    # We can compute this by:
    # for i in 1..N:
    #   for j in i, 2i, 3i... <= N:
    #     sigma[j] += i
    # This is O(N log N). For N=10^7, this is roughly 10^7 * 16 = 1.6 * 10^8 operations.
    # In numpy, we can vectorize the inner loop.
    
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() > hard_stop - 10.0:
        return [11, 6]

    # 2. Group numbers by their sigma value.
    # We want to find pairs (a, b) with sigma(a) == sigma(b) minimizing |a/b - 1.5|.
    # Let's create a dictionary mapping sigma value -> list of numbers.
    # To save memory, we can store the list of numbers for each sigma value.
    # Note: Many sigma values are unique. We only care about those with >= 2 numbers.
    
    # Instead of a dict of lists (which can be slow to build), let's sort by sigma value.
    # indices = np.arange(1, N+1)
    # sorted_indices = indices[np.argsort(sigma[1:])]
    # This sorts 10^7 elements. Should be fast enough (few seconds).
    
    idx = np.arange(1, N + 1, dtype=np.int32)
    vals = sigma[1:]
    order = np.argsort(vals, kind='stable')
    svals = vals[order]
    
    if time.time() > hard_stop - 8.0:
        return [11, 6]

    # Identify groups of same sigma value
    # A group starts where svals changes.
    M = order.size
    change = np.empty(M, dtype=bool)
    change[0] = True
    change[1:] = svals[1:] != svals[:-1]
    starts = np.nonzero(change)[0]
    ends = np.append(starts[1:], M)
    
    best_error = 1.0  # Worse than seed's 1/3
    best_pair = (11, 6)
    
    # Iterate through groups
    # We want to find a, b in the same group minimizing |2a - 3b| / (2b).
    # This is equivalent to minimizing |2a - 3b| for a fixed b, or generally finding close ratios.
    # Since the group is sorted by sigma, not by value, the numbers in the group are not sorted by value.
    # We need to sort the numbers within each group to use binary search or two-pointer techniques.
    
    for gi in range(starts.size):
        if time.time() > hard_stop - 2.0:
            break
            
        s0 = int(starts[gi])
        e0 = int(ends[gi])
        length = e0 - s0
        
        if length < 2:
            continue
            
        # Extract numbers in this group
        # gn = np.sort(idx[order[s0:e0]])
        # Sorting small lists is fast.
        gn = idx[order[s0:e0]]
        gn_sorted = np.sort(gn).tolist()
        
        # For each b in gn_sorted, find a in gn_sorted such that a/b is close to 1.5.
        # Target a = 1.5 * b.
        # We can use bisect to find the position of 1.5*b in gn_sorted.
        L = len(gn_sorted)
        
        # Optimization: If the group is very large, we might spend too much time.
        # But most groups are small.
        
        for i in range(L):
            b = gn_sorted[i]
            target = 1.5 * b
            
            # Find insertion point
            pos = bisect_left(gn_sorted, target)
            
            candidates = []
            if pos < L:
                candidates.append(gn_sorted[pos])
            if pos > 0:
                candidates.append(gn_sorted[pos - 1])
                
            for a in candidates:
                if a == b:
                    continue
                # Calculate error
                # |a/b - 1.5| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                if num == 0:
                    # Exact match is invalid per task ("raw value of exactly 0 cannot be scored")
                    # But wait, the task says "Minimize |a/b-3/2|". 
                    # "A raw value of exactly 0 cannot be scored and counts as invalid."
                    # So we must avoid a/b == 3/2 exactly.
                    continue
                den = 2 * b
                err = num / den
                
                if err < best_error:
                    best_error = err
                    best_pair = (a, b)
                    
    a, b = best_pair
    a = int(a)
    b = int(b)
    
    # Final validation
    if a == b or sigma[a] != sigma[b] or a < 1 or b < 1 or a > N or b > N:
        return [11, 6]
    if 2 * a == 3 * b:
        return [11, 6]
        
    return [a, b]

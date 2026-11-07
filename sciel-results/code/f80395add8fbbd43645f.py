import numpy as np
import time
from bisect import bisect_left
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 10000000
    
    # 1. Compute sigma(n) for all n up to N using a more efficient approach
    # sigma(n) = sum of all positive divisors of n
    # Use a sieve-based approach with numpy for speed
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # For each divisor d, add d to all multiples of d
    # This is O(N log N) but vectorized
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    # Check time after sieve
    if time.time() > deadline:
        # Return a known valid pair as fallback
        return [11, 6]

    # 2. Group numbers by sigma value
    # Use defaultdict for efficiency
    groups = defaultdict(list)
    for n in range(1, N + 1):
        s = int(sigma[n])
        groups[s].append(n)

    # Check time after grouping
    if time.time() > deadline:
        return [11, 6]

    # 3. Find best pair minimizing |a/b - 3/2| = |2a - 3b| / (2b)
    # For each group, we need to find a, b in the group that minimizes this
    # Since all values in a group are sorted, we can use a two-pointer approach
    # But we need to be careful: we want to minimize |2a - 3b|/(2b)
    
    best_score = float('inf')
    best_a, best_b = 11, 6  # Default valid pair: sigma(11)=12, sigma(6)=12
    
    # For efficiency, only process groups with at least 2 elements
    for s, L in groups.items():
        if len(L) < 2:
            continue
        
        # L is already sorted (we appended in order)
        # We want to find a, b in L minimizing |2a - 3b| / (2b)
        # This is equivalent to finding a/b closest to 3/2
        # Use binary search for each b to find the best a
        
        n_l = len(L)
        # For each b in L, find a in L that makes a/b closest to 1.5
        # i.e., a closest to 1.5*b
        
        for j in range(n_l):
            b = L[j]
            target_a = 1.5 * b
            
            # Find the position where target_a would be in L
            # We want a values close to target_a
            pos = bisect_left(L, int(target_a))
            
            # Check positions around pos
            for check_pos in [pos - 1, pos, pos + 1]:
                if 0 <= check_pos < n_l:
                    a = L[check_pos]
                    if a == b:
                        continue
                    # Compute |a/b - 3/2| = |2a - 3b| / (2b)
                    diff = abs(2 * a - 3 * b)
                    if diff == 0:
                        # This would give score 0, which is invalid
                        # Skip this pair
                        continue
                    current_score = diff / (2.0 * b)
                    if current_score < best_score:
                        best_score = current_score
                        best_a = a
                        best_b = b
        
        # Check time periodically
        if time.time() > deadline:
            break

    return [int(best_a), int(best_b)]

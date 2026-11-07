import time
import math
import numpy as np
from fractions import Fraction
import bisect

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    L = 10_000_000
    
    # 1. Compute sigma(n) for n=1..L
    sigma = np.zeros(L + 1, dtype=np.int64)
    for i in range(1, L + 1):
        sigma[i::i] += i
        
    # 2. Build mapping from sigma value to list of n
    # Note: sigma values can be large, but number of unique values is manageable
    sigma_map = {}
    for n in range(1, L + 1):
        s = int(sigma[n])
        if s not in sigma_map:
            sigma_map[s] = []
        sigma_map[s].append(n)
        
    # 3. Sort lists for binary search
    for s in sigma_map:
        sigma_map[s].sort()
        
    # 4. Search for best pair
    best_error = float('inf')
    best_a = 1
    best_b = 1
    best_num = 1
    best_den = 1
    
    # Iterate b from L down to 1
    # We want to minimize |2a - 3b| / (2b)
    # Lower bound on error for a given b is 1/(2b) (if 2a-3b = +/- 1)
    # So if 1/(2b) >= best_error, we can stop.
    # i.e., 2b <= 1/best_error => b <= 1/(2*best_error)
    
    # Initial best error is inf, so we start at L
    # As we find better errors, the threshold for b decreases
    
    for b in range(L, 0, -1):
        if time.time() - start_time > deadline - 2:
            break
            
        # Early termination check
        # If 1/(2*b) >= best_error, then for this b and all smaller b, 
        # the minimum possible error is >= best_error.
        # So we can break.
        if best_error > 0 and (1.0 / (2.0 * b)) >= best_error:
            break
            
        s = int(sigma[b])
        candidates = sigma_map.get(s)
        if not candidates or len(candidates) < 2:
            continue
            
        # We need a != b
        # Target a is approx 1.5 * b
        target_a = 1.5 * b
        
        # Find closest a in candidates
        # Use bisect to find insertion point
        idx = bisect.bisect_left(candidates, target_a)
        
        # Check neighbors: idx-1, idx, idx+1, idx+2 (to be safe with ties or close values)
        # Also check idx-2
        # Since we want a != b, we skip if a == b
        
        checked_indices = set()
        for delta in [-2, -1, 0, 1, 2]:
            i = idx + delta
            if 0 <= i < len(candidates):
                if i in checked_indices:
                    continue
                checked_indices.add(i)
                a = candidates[i]
                if a == b:
                    continue
                
                # Calculate error exactly
                # Error = |a/b - 3/2| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                if num == 0:
                    # This is invalid per task: "raw value of exactly 0 cannot be scored"
                    continue
                den = 2 * b
                err = num / den
                
                if err < best_error:
                    best_error = err
                    best_a = a
                    best_b = b
                    best_num = num
                    best_den = den
                    
    # Return as list of Python ints
    return [int(best_a), int(best_b)]

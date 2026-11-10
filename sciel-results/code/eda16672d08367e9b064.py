import time
import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    LCM = 30030  # lcm of all divisors

    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s wall-clock kill

    # We iterate W from the theoretical minimum (n-1=9) upwards
    # The span must be at least n-1 because numbers are distinct integers.
    
    # Pre-compute divisors sorted by some heuristic? 
    # Actually, for the matching, the order of divisors doesn't matter for existence, 
    # but for backtracking speed, we should process the most constrained ones first.
    
    for W in range(9, 500):
        if time.time() > deadline:
            break
            
        # If W is small, use bitmask for speed
        use_bitmask = W < 64
        
        for L in range(LCM):
            if time.time() > deadline:
                break
            
            # Compute candidates for each divisor
            # candidates[i] is a list of offsets r in [0, W] such that (L+r) % divisors[i] == 0
            candidates = []
            valid = True
            for d in divisors:
                r0 = (-L) % d
                if r0 > W:
                    valid = False
                    break
                cands = []
                r = r0
                while r <= W:
                    cands.append(r)
                    r += d
                if not cands:
                    valid = False
                    break
                candidates.append(cands)
            
            if not valid:
                continue
            
            # MRV ordering
            indices = list(range(n))
            indices.sort(key=lambda i: len(candidates[i]))
            
            # Quick Hall's check for top k
            # If union of candidates for any subset of size k is < k, no match.
            # We check for k=1..min(n, 4) on the sorted indices
            for k in range(1, min(n, 4) + 1):
                union = set()
                for idx in range(k):
                    union.update(candidates[indices[idx]])
                if len(union) < k:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Backtracking
            # We want to assign an offset r to each divisor i.
            # All r must be distinct.
            
            if use_bitmask:
                used_mask = 0
                choices = [-1] * n
                
                def backtrack_bitmask(idx):
                    nonlocal used_mask
                    if idx == n:
                        return True
                    
                    i = indices[idx]
                    # Try candidates for divisor i
                    for r in candidates[i]:
                        bit = 1 << r
                        if not (used_mask & bit):
                            used_mask |= bit
                            choices[i] = r
                            if backtrack_bitmask(idx + 1):
                                return True
                            used_mask &= ~bit
                            choices[i] = -1
                    return False
                
                if backtrack_bitmask(0):
                    result = []
                    for i in range(n):
                        result.append(L + choices[i])
                    # Verify
                    if len(set(result)) != n:
                        continue
                    for i, x in enumerate(result):
                        if x % divisors[i] != 0 or x <= 0 or x > 1000000:
                            break
                    else:
                        return [int(x) for x in result]
            else:
                used = [False] * (W + 1)
                choices = [-1] * n
                
                def backtrack_list(idx):
                    if idx == n:
                        return True
                    
                    i = indices[idx]
                    for r in candidates[i]:
                        if not used[r]:
                            used[r] = True
                            choices[i] = r
                            if backtrack_list(idx + 1):
                                return True
                            used[r] = False
                            choices[i] = -1
                    return False
                
                if backtrack_list(0):
                    result = []
                    for i in range(n):
                        result.append(L + choices[i])
                    if len(set(result)) != n:
                        continue
                    for i, x in enumerate(result):
                        if x % divisors[i] != 0 or x <= 0 or x > 1000000:
                            break
                    else:
                        return [int(x) for x in result]

    # Fallback if not found (should not happen)
    return [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]

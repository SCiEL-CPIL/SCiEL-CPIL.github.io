import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin before 90s kill
    
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Compute LCM
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for d in divisors:
        L = lcm(L, d)
    # L = 30030
    
    best_span = float('inf')
    best_result = None
    
    # For each window size W, try to find a valid assignment
    # Lower bound: W >= 9 (need 10 distinct integers)
    # Also, for each divisor d_i, the window must contain at least one multiple of d_i
    
    for W in range(9, 200):  # Search up to W=199
        if time.time() > deadline:
            break
        
        # For each starting position s in one period
        for s in range(L):
            if time.time() > deadline:
                break
            
            # For each divisor, find all multiples in [s, s+W]
            candidates = []
            valid = True
            for i, d in enumerate(divisors):
                # First multiple of d >= s
                if s == 0:
                    first = d
                else:
                    first = ((s + d - 1) // d) * d
                if first > s + W:
                    valid = False
                    break
                # All multiples: first, first+d, first+2d, ... up to s+W
                cands = []
                m = first
                while m <= s + W:
                    cands.append(m)
                    m += d
                if not cands:
                    valid = False
                    break
                candidates.append(cands)
            
            if not valid:
                continue
            
            # Now try to assign distinct values from candidates
            # Use backtracking
            assignment = [-1] * n
            used = set()
            
            def backtrack(idx):
                if idx == n:
                    return True
                for val in candidates[idx]:
                    if val not in used:
                        used.add(val)
                        assignment[idx] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                        assignment[idx] = -1
                return False
            
            if backtrack(0):
                span = W
                if span < best_span:
                    best_span = span
                    best_result = [int(v) for v in assignment]
                    # We found a solution with span W, can't do better than W
                    # But we want the minimum, so we should continue searching smaller W
                    # Actually we're iterating W from small to large, so first found is best
                    # Wait, we need to make sure all values are <= 1000000
                    max_val = max(assignment)
                    if max_val <= 1000000:
                        return [int(v) for v in assignment]
                    else:
                        # Need to check if we can shift to make it fit
                        # Since the pattern repeats every L=30030, we can shift by k*L
                        # But shifting changes the actual values
                        # The values are already specific, so if max > 1000000, this s won't work
                        # Try next s
                        pass
            else:
                pass
    
    if best_result is not None:
        return [int(v) for v in best_result]
    
    # Fallback: just return the divisors themselves (valid but not optimal)
    return [int(d) for d in divisors]

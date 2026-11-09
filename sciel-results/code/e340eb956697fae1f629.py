import time
import sys
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    deadline = time.time() + 55.0
    
    lcm_val = reduce(_lcm, divisors)
    
    best_spread = float('inf')
    best_values = None
    
    # Helper to check if a valid assignment exists in [L, L+W]
    def check_assignment(L, W):
        cands = []
        for d in divisors:
            start = ((L + d - 1) // d) * d
            if start > L + W:
                return None
            c = []
            x = start
            while x <= L + W:
                c.append(x)
                x += d
            cands.append(c)
            
        if any(len(c) == 0 for c in cands):
            return None
            
        # Order by number of candidates
        order = sorted(range(n), key=lambda i: len(cands[i]))
        used = set()
        assigned = [None] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            orig_idx = order[idx]
            for val in cands[orig_idx]:
                if val not in used:
                    used.add(val)
                    assigned[orig_idx] = val
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
            return False
            
        if backtrack(0):
            return assigned
        return None

    # Iterate L
    # We can limit L to LCM
    for L in range(1, lcm_val + 1):
        if time.time() > deadline - 2.0:
            break
            
        # Binary search on W
        # Max W needed? The divisors themselves have spread 33.
        # So W=33 is always a valid upper bound (using divisors shifted to start near L? No, divisors are fixed values).
        # But we can always pick multiples.
        # A safe upper bound for W is 100.
        lo = 0
        hi = 100
        
        # Check if hi is valid
        res_hi = check_assignment(L, hi)
        if res_hi is None:
            # Try larger hi? Unlikely for small L.
            # If no solution for W=100, skip this L.
            continue
            
        # Binary search for minimal W
        while lo < hi:
            mid = (lo + hi) // 2
            res_mid = check_assignment(L, mid)
            if res_mid is not None:
                hi = mid
            else:
                lo = mid + 1
                
        # lo is the minimal W for this L
        if lo < best_spread:
            best_spread = lo
            # Retrieve the values
            res_best = check_assignment(L, lo)
            if res_best:
                best_values = res_best
                
            if best_spread == 9:
                break
                
    if best_values is None:
        return [int(d) for d in divisors]
        
    return [int(v) for v in best_values]

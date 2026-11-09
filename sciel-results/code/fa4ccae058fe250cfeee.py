"""Rewritten program: SEARCH approach with periodicity and backtracking."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

P = {
    'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39],
    'maximum': 1000000
}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    MAX_V = P['maximum']
    
    # Calculate LCM of all divisors
    def lcm(a, b):
        return a * b // gcd(a, b)
    lcm_all = reduce(lcm, divisors)
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Iterate span W from minimum possible (9) upwards
    # We don't know the upper bound, but it should be small.
    # Let's try up to 200 first.
    for W in range(9, 300):
        if time.time() > deadline:
            break
            
        # We only need to check L in [1, lcm_all] due to periodicity
        # Also ensure L+W <= MAX_V
        max_L = min(lcm_all, MAX_V - W)
        
        for L in range(1, max_L + 1):
            if time.time() > deadline:
                break
                
            # Generate candidates for each divisor
            candidates = []
            feasible = True
            for d in divisors:
                # First multiple >= L
                first_k = (L + d - 1) // d
                # Last multiple <= L+W
                last_k = (L + W) // d
                
                if first_k > last_k:
                    feasible = False
                    break
                
                cands = [k * d for k in range(first_k, last_k + 1)]
                candidates.append(cands)
                
            if not feasible:
                continue
            
            # Quick check: if any candidate list is empty, infeasible (already handled)
            # Also, if the union of all candidates has size < n, infeasible
            # This is a cheap check
            all_cands = set()
            for c in candidates:
                all_cands.update(c)
            if len(all_cands) < n:
                continue
            
            # Backtracking with MRV (Minimum Remaining Values)
            # Order divisors by number of candidates
            indices = list(range(n))
            indices.sort(key=lambda i: len(candidates[i]))
            
            assigned = [0] * n
            used = set()
            
            def backtrack(idx):
                if idx == n:
                    return True
                
                i = indices[idx]
                for val in candidates[i]:
                    if val not in used:
                        used.add(val)
                        assigned[i] = val
                        if backtrack(idx + 1):
                            return True
                        used.discard(val)
                        assigned[i] = 0
                return False
            
            if backtrack(0):
                # Verify and return
                # The assignment is in 'assigned', but we need to return in original divisor order
                # 'assigned' is already indexed by original divisor index i
                result = [int(v) for v in assigned]
                
                # Double check constraints
                if len(set(result)) != n:
                    continue
                for i, d in enumerate(divisors):
                    if result[i] % d != 0:
                        continue
                if any(v < 1 or v > MAX_V for v in result):
                    continue
                
                return result
    
    # Fallback if not found within W < 300 or time limit
    # This should not happen for this problem, but ensures a valid return
    # A simple valid solution: x_i = d_i * K for large enough K to be distinct?
    # Actually, d_i are distinct, so x_i = d_i works if they are distinct.
    # 6, 10, 14, 15, 21, 22, 26, 33, 35, 39 are all distinct.
    # So returning divisors is always valid.
    return [int(d) for d in divisors]

# EVOLVE-BLOCK-END

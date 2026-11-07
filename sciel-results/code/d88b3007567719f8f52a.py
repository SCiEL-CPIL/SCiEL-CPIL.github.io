import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Compute LCM
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for d in divisors:
        L = lcm(L, d)
    # L = 30030
    
    # We search for the minimum span S
    # Lower bound is 9 (10 distinct integers)
    # Upper bound: The divisors themselves form a valid set with span 39-6=33.
    # So we only need to search S from 9 to 33.
    # If no solution is found in [9, 33], we return the divisors (span 33).
    
    for S in range(9, 34):
        if time.time() > deadline:
            break
        
        for s in range(L):
            if time.time() > deadline:
                break
            
            # 1. Quick feasibility check: 
            # For each divisor d_i, the distance to the next multiple must be <= S
            feasible = True
            for i, d in enumerate(divisors):
                r = s % d
                # Distance to next multiple
                # If r == 0, dist = 0
                # If r > 0, dist = d - r
                dist = (d - r) % d
                if dist > S:
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            # 2. Build candidate lists
            candidates = []
            for i, d in enumerate(divisors):
                r = s % d
                if r == 0:
                    first = s
                else:
                    first = s + (d - r)
                
                # Ensure positive integer
                if first == 0:
                    first = d
                    
                cands = []
                m = first
                while m <= s + S:
                    cands.append(m)
                    m += d
                candidates.append(cands)
            
            # 3. Backtracking with MRV (Minimum Remaining Values)
            # Order indices by the number of candidates
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            assignment = [-1] * n
            used = set()
            
            def backtrack(idx):
                if idx == n:
                    return True
                i = order[idx]
                for val in candidates[i]:
                    if val not in used:
                        used.add(val)
                        assignment[i] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                        assignment[i] = -1
                return False
            
            if backtrack(0):
                # Found a valid assignment with span S
                # Check if all values are <= 1000000
                # Since s < 30030 and S <= 33, max value < 30063, which is < 1000000
                return [int(v) for v in assignment]
        
        # If we finish the s-loop for this S without finding a solution, 
        # we continue to the next S.
    
    # Fallback: Return the divisors themselves
    # This is always valid and has span 33
    return [int(d) for d in divisors]

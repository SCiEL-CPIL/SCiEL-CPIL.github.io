import time
import math
from math import gcd
from functools import reduce

# Pinned input
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Calculate LCM of all divisors to exploit periodicity
    def lcm(a, b):
        return a * b // gcd(a, b)
    M = reduce(lcm, DIVISORS)  # M = 30030
    
    best_span = float('inf')
    best_assign = None
    
    # The minimum possible span is N-1 = 9 (10 distinct integers)
    # We iterate W from 9 upwards
    for W in range(9, 100):
        if time.time() > deadline:
            break
            
        found_for_w = False
        # Iterate through all possible start positions in one period
        for L in range(M):
            if time.time() > deadline:
                break
                
            # Check if a valid assignment exists for window [L, L+W]
            # Candidates for divisor i: multiples of DIVISORS[i] in [L, L+W]
            candidates = []
            valid = True
            for i, d in enumerate(DIVISORS):
                # Find k such that L <= k*d <= L+W
                k_min = (L + d - 1) // d
                k_max = (L + W) // d
                
                if k_min > k_max:
                    valid = False
                    break
                
                vals = [k * d for k in range(k_min, k_max + 1)]
                candidates.append(vals)
            
            if not valid:
                continue
            
            # Solve bipartite matching / SDR using backtracking with MRV
            # Sort indices by number of candidates (most constrained first)
            order = sorted(range(N), key=lambda i: len(candidates[i]))
            
            used = set()
            assignment = [None] * N
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == N:
                    return True
                
                # MRV: pick unassigned variable with smallest available domain
                min_avail = float('inf')
                best_i = -1
                for j in range(N):
                    if assignment[j] is not None:
                        continue
                    avail = 0
                    for v in candidates[j]:
                        if v not in used:
                            avail += 1
                            if avail >= min_avail:
                                break
                    if avail == 0:
                        return False
                    if avail < min_avail:
                        min_avail = avail
                        best_i = j
                
                if best_i == -1:
                    return True
                
                # Try values for best_i
                for v in candidates[best_i]:
                    if v in used:
                        continue
                    used.add(v)
                    assignment[best_i] = v
                    
                    if backtrack(idx + 1):
                        return True
                        
                    used.remove(v)
                    assignment[best_i] = None
                
                return False
            
            if backtrack(0):
                # Found a valid set of values for this L and W
                # We need to ensure these values can be shifted to [1, MAX_VAL]
                v_min = min(assignment)
                v_max = max(assignment)
                
                # Find t such that 1 <= v_i + t*M <= MAX_VAL for all i
                # t_min = ceil((1 - v_min) / M)
                # t_max = floor((MAX_VAL - v_max) / M)
                
                t_min = 0
                if v_min < 1:
                    t_min = (1 - v_min + M - 1) // M
                
                t_max = (MAX_VAL - v_max) // M
                
                if t_min <= t_max:
                    t = t_min
                else:
                    continue
                
                # Apply shift
                final_assign = [assignment[i] + t * M for i in range(N)]
                
                # Verify
                for i, d in enumerate(DIVISORS):
                    if final_assign[i] < 1 or final_assign[i] > MAX_VAL:
                        break
                    if final_assign[i] % d != 0:
                        break
                    if len(set(final_assign)) != N:
                        break
                else:
                    best_span = W
                    best_assign = final_assign
                    found_for_w = True
                    break
        
        if found_for_w:
            break
    
    # If no solution found within time or W limit, return fallback
    if best_assign is None:
        # Fallback: just the divisors themselves (span 33)
        best_assign = DIVISORS[:]
    
    # Final type checking and validation
    result = [int(v) for v in best_assign]
    
    return result

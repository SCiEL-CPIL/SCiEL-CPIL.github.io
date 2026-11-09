"""Rewritten program: MODEL-AND-OPTIMIZE approach to pack distinct multiples of fixed divisors."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

# Pinned input
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = len(DIVISORS)
    
    # Calculate LCM of all divisors to exploit periodicity
    def lcm(a, b):
        return a * b // gcd(a, b)
    M = reduce(lcm, DIVISORS)  # LCM = 30030
    
    best_span = float('inf')
    best_assign = None
    
    # Quick check: does [L, L+W] contain at least one multiple of d?
    def has_multiple(L, W, d):
        # Number of multiples of d in [L, L+W]
        if L == 0:
            return (L + W) // d >= 1
        else:
            return (L + W) // d - (L - 1) // d >= 1
    
    # Feasibility check with quick pruning
    def check_window(L, W):
        """Check if window [L, L+W] contains a valid assignment. Returns list of values or None."""
        # Quick prune: check each divisor has at least one multiple
        candidate_lists = []
        for d in DIVISORS:
            if not has_multiple(L, W, d):
                return None
            first = (-L) % d
            offsets = []
            for j in range(first, W + 1, d):
                offsets.append(j)
            candidate_lists.append(offsets)
        
        # Sort by number of candidates (most constrained first)
        order = sorted(range(n), key=lambda i: len(candidate_lists[i]))
        
        used = set()
        assignment = [0] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            for j in candidate_lists[i]:
                if j not in used:
                    used.add(j)
                    assignment[i] = j
                    if backtrack(idx + 1):
                        return True
                    used.remove(j)
            return False
        
        if not backtrack(0):
            return None
        
        return [L + assignment[i] for i in range(n)]
    
    # Iterate W from 9 (minimum for 10 distinct ints) upward
    for W in range(9, 30):
        if time.time() > deadline:
            break
        
        found_for_w = False
        # Due to periodicity, we only need L in [0, M-1]
        for L in range(M):
            if time.time() > deadline:
                break
            
            result = check_window(L, W)
            if result is not None:
                # Found a solution! Now shift to fit in [1, MAX_VAL]
                v_min = min(result)
                v_max = max(result)
                
                # Find t such that 1 <= v_i + t*M <= MAX_VAL
                t_min = 0
                if v_min < 1:
                    t_min = (1 - v_min + M - 1) // M
                t_max = (MAX_VAL - v_max) // M
                
                if t_min <= t_max:
                    t = t_min
                    final_assign = [v + t * M for v in result]
                    
                    # Verify
                    valid = True
                    for i, d in enumerate(DIVISORS):
                        if final_assign[i] < 1 or final_assign[i] > MAX_VAL:
                            valid = False
                            break
                        if final_assign[i] % d != 0:
                            valid = False
                            break
                    if len(set(final_assign)) != n:
                        valid = False
                    
                    if valid:
                        best_span = W
                        best_assign = final_assign
                        found_for_w = True
                        break
        
        if found_for_w:
            break
    
    # If no solution found within time or W limit, return fallback
    if best_assign is None:
        # Fallback: just the divisors themselves
        best_assign = DIVISORS[:]
    
    # Final type checking and validation
    result = [int(v) for v in best_assign]
    
    for i, d in enumerate(DIVISORS):
        assert isinstance(result[i], int)
        assert not isinstance(result[i], bool)
        assert 1 <= result[i] <= MAX_VAL
        assert result[i] % d == 0
        
    assert len(set(result)) == n
    
    return result

# EVOLVE-BLOCK-END

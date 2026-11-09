import time
import bisect
from math import gcd
from functools import reduce

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    max_val = P['maximum']
    n = len(divisors)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0

    # Precompute multiples for each divisor
    multiples = {}
    for d in divisors:
        multiples[d] = list(range(d, max_val + 1, d))

    # We know a solution with range 16 exists (or at least range 33).
    # Let's try to find a solution with range < 16.
    # Minimum possible range is 9 (10 distinct ints).
    
    # Helper to solve for a specific window [L, L+W]
    def solve_window(L, W):
        # Get candidates for each divisor
        candidates = []
        for d in divisors:
            m_list = multiples[d]
            # Find indices of multiples in [L, L+W]
            left_idx = bisect.bisect_left(m_list, L)
            right_idx = bisect.bisect_right(m_list, L + W)
            
            if left_idx >= right_idx:
                return None
            
            # Extract candidates
            cand = m_list[left_idx:right_idx]
            candidates.append(cand)
        
        # Backtracking solver with MRV
        # State: assignment[i] = value, used = set of values
        assignment = [-1] * n
        used = set()
        
        # Order indices by number of candidates (MRV)
        # We re-sort at each step? No, static order based on initial domain size is usually fine for small N,
        # but dynamic MRV is better. Given N=10, static is likely sufficient and faster.
        order = sorted(range(n), key=lambda i: len(candidates[i]))
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == n:
                return True
            
            var_idx = order[idx]
            
            for val in candidates[var_idx]:
                if val not in used:
                    used.add(val)
                    assignment[var_idx] = val
                    
                    if backtrack(idx + 1):
                        return True
                    
                    used.remove(val)
                    assignment[var_idx] = -1
        
        if backtrack(0):
            return assignment
        return None

    # Search for minimal W
    # We start from W=10 up to 15.
    # For each W, we search L.
    # How far should we search L?
    # The pattern of multiples repeats every LCM=3003000.
    # But for small W, valid L's are dense.
    # Let's search L from 1 to 2000 for each W.
    # If not found, we might need to go further, but let's try this first.
    
    # If we don't find anything < 16, we should return a known good solution.
    # Let's find a solution for W=16 first as a fallback?
    # Or just use the divisors themselves (Range 33).
    # Actually, the current program found 16. Let's try to find 15, 14, etc.
    
    # Fallback solution: The divisors themselves
    fallback = divisors[:]
    fallback_range = max(fallback) - min(fallback)
    
    best_solution = None
    best_range = fallback_range
    
    # Try W from 10 to 15
    for W in range(10, 16):
        if time.time() > deadline:
            break
            
        # Search L
        # We can optimize by stepping L.
        # If a solution exists for L, it might not exist for L+1.
        # We must check each L.
        
        # Limit L search to avoid timeout.
        # If W is small, the "density" of valid L's is high.
        # Let's check L from 1 to 5000.
        max_L_check = 5000
        
        for L in range(1, max_L_check + 1):
            if time.time() > deadline:
                break
                
            sol = solve_window(L, W)
            if sol is not None:
                # Verify distinctness and constraints (though solver ensures it)
                # Check if this is better than current best
                if W < best_range:
                    best_range = W
                    best_solution = sol[:]
                    # If we found a very good solution, we can stop early?
                    # No, we want the minimum W. If we find W=10, that's optimal.
                    # But we iterate W upwards, so the first found is the best for that W.
                    # We break out of L loop, and continue to next W?
                    # No, if we find a solution for W, it is the global minimum W found so far.
                    # Since we iterate W from small to large, the first W with a solution is the answer.
                    return sol
        
        # If no solution found for this W in the limited L range, 
        # it's possible a solution exists for larger L.
        # But given the density, it's unlikely for small W.
        # If we reach here, we continue to next W.

    # If we didn't find anything < 16, return the fallback (or a known 16 solution if we had one)
    # The prompt says current score is 2.0625, implying range 16.
    # We should try to find a range 16 solution if we didn't find < 16.
    # Let's do one more pass for W=16 with a slightly larger L range if we have time.
    if time.time() < deadline - 5.0:
        W = 16
        max_L_check = 2000
        for L in range(1, max_L_check + 1):
            if time.time() > deadline:
                break
            sol = solve_window(L, W)
            if sol is not None:
                return sol

    return fallback

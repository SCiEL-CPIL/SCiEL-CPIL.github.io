import time
import sys

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    deadline = time.time() + 55.0
    
    # We search for the minimum W
    # L can be any positive integer.
    # Due to the nature of multiples, a solution for L implies a solution for L + K*LCM(divisors)
    # But we just need one. Small L is likely to work.
    
    # Heuristic: The window size W must be at least large enough to accommodate
    # the "tightest" constraints.
    # Let's iterate W from 1 to 1000.
    # For each W, iterate L from 1 to 1000.
    # If a solution is found, return it.
    
    # To make it faster, we can pre-calculate candidates for a given L, W?
    # No, L changes.
    
    # Optimization: 
    # For a fixed W, the condition "divisor d has a multiple in [L, L+W]"
    # depends on L mod d.
    # We can iterate L more smartly, but simple iteration is likely fast enough
    # because the first valid W will be small.
    
    best_result = None
    best_W = float('inf')
    
    # We will try W from 1 upwards.
    # The max W is bounded by 1000000, but realistically < 100.
    
    # We need to ensure we don't timeout.
    # If we don't find a solution quickly, we might need to expand L range.
    # But let's assume small L works.
    
    # Helper to solve assignment for given L, W
    def solve_assignment(L, W):
        # Candidates for each divisor
        cands = []
        for d in divisors:
            # Find multiples of d in [L, L+W]
            # First multiple >= L
            start = ((L + d - 1) // d) * d
            if start > L + W:
                return None
            # Generate candidates
            c = []
            x = start
            while x <= L + W:
                c.append(x)
                x += d
            cands.append(c)
        
        # Check if any candidate list is empty
        if any(len(c) == 0 for c in cands):
            return None
        
        # Sort divisors by number of candidates (ascending) to prune early
        # We need to keep track of original indices to return in correct order
        order = sorted(range(n), key=lambda i: len(cands[i]))
        
        assigned = [None] * n
        used_values = set()
        
        def backtrack(idx):
            if idx == n:
                return True
            orig_idx = order[idx]
            for val in cands[orig_idx]:
                if val not in used_values:
                    used_values.add(val)
                    assigned[orig_idx] = val
                    if backtrack(idx + 1):
                        return True
                    used_values.remove(val)
            return False
        
        if backtrack(0):
            return assigned
        return None

    # Search loop
    # We can limit L to a certain range. 
    # If no solution found for L in 1..100 for a given W, 
    # it's unlikely to be the minimal W if a smaller W didn't work either?
    # Not necessarily. But let's try L up to 1000 for each W.
    
    # To be safe against timeout, we track time.
    # If we are close to deadline, we break and return best found so far.
    # But we are looking for the FIRST valid W, which is the minimum.
    # So we just need to find ANY valid configuration for the current W.
    
    for W in range(1, 200):
        if time.time() > deadline - 5.0:
            break
            
        # Try different L values
        # We can optimize L search by noticing that the pattern of multiples repeats every LCM(divisors).
        # LCM of 6,10,14,15,21,22,26,33,35,39
        # 6=2*3, 10=2*5, 14=2*7, 15=3*5, 21=3*7, 22=2*11, 26=2*13, 33=3*11, 35=5*7, 39=3*13
        # Primes: 2,3,5,7,11,13
        # LCM = 2 * 3 * 5 * 7 * 11 * 13 = 30030
        # So L mod 30030 determines the pattern.
        # We only need to check L in [1, 30030].
        # But 30030 * 200 (W) is 6 million checks. Too slow.
        
        # However, for small W, the "gap" between multiples is large relative to W?
        # No, W is small.
        # Let's just check L from 1 to 1000.
        # If we don't find a solution, we might miss the optimal L for that W.
        # But the optimal W will likely have a solution for small L.
        
        found = False
        for L in range(1, 1001):
            if time.time() > deadline - 2.0:
                break
            res = solve_assignment(L, W)
            if res is not None:
                # Verify distinctness and constraints
                if len(set(res)) == n and all(1 <= x <= max_val for x in res):
                    # Check divisibility
                    valid = True
                    for i in range(n):
                        if res[i] % divisors[i] != 0:
                            valid = False
                            break
                    if valid:
                        return [int(x) for x in res]
                else:
                    # Should not happen if solve_assignment is correct
                    pass
        # If no L in 1..1000 worked for this W, try next W
        # It is possible that a solution exists for larger L.
        # But we are minimizing W. If W is small, L should be small.
        
    # Fallback: return a valid (but not necessarily optimal) solution
    # Just return the divisors themselves (which are distinct and valid)
    # Wait, divisors are [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    # Max - Min = 39 - 6 = 33.
    # This is a valid solution.
    # But we want to minimize.
    # If the search failed, return the divisors.
    
    return [int(d) for d in divisors]

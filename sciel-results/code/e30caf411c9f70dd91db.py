"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = 10
MAX_VAL = 1000000

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    divisors = DIVISORS[:]
    n = len(divisors)
    
    # Compute LCM of all divisors
    lcm_val = 1
    for d in divisors:
        lcm_val = _lcm(lcm_val, d)
    
    # Trivial solution
    trivial = divisors[:]
    best_R = max(trivial) - min(trivial)
    best_result = trivial
    
    # Search for smaller R using Bitmask DP
    # For a window [S, S+R], each divisor d has some multiples in the window.
    # We need to assign one distinct multiple to each divisor.
    # This is a bipartite matching problem, which can be solved with Bitmask DP.
    
    # For each R from 9 upward:
    #   For each S from 0 to lcm_val-1:
    #     Compute for each divisor i, the set of offsets (relative to S) where multiples fall
    #     Check if perfect matching exists using DP over subsets of divisors
    
    # Optimization: For a given R, precompute for each divisor d and each residue s mod d,
    # the list of offsets in [0, R] that are multiples of d starting from s.
    # Actually, for window [S, S+R], the multiples of d are at positions:
    #   k*d where S <= k*d <= S+R
    #   So k ranges from ceil(S/d) to floor((S+R)/d)
    #   The actual values are k*d, which relative to S are k*d - S.
    
    # For DP: we have n divisors. For each divisor i, we have a list of candidate values.
    # We want to pick one value from each list, all distinct.
    # DP state: (mask, last_value) where mask is which divisors we've assigned,
    # and last_value is... hmm, this doesn't work directly because values can be large.
    
    # Alternative: since R is small (starting from 9), the candidate values are in [S, S+R],
    # so there are at most R+1 possible distinct values. We can use a different DP:
    # DP over the values in the window, deciding which divisor gets which value.
    
    # Actually, the standard approach for this small problem:
    # For each S, build a bipartite graph between divisors and values in [S, S+R].
    # Use Hopcroft-Karp or simple matching. But for n=10 and window size <= 20,
    # a simple DFS-based matching is fast enough.
    
    # Let's use the bitmask DP approach properly:
    # For each S, collect for each divisor i the set of candidate values (as offsets from S, 0..R).
    # Then use DP: dp[mask] = set of values used, or better, dp[mask] = True/False
    # but we need to track which values are used. Since values are 0..R (at most ~20),
    # we can use a different formulation.
    
    # Actually, let's just use backtracking with MRV for each (R, S) pair,
    # but with aggressive pruning and early termination.
    
    # Better: for each R, iterate S. For each S, check feasibility.
    # Use a fast check: build the candidate lists, then use recursive matching with MRV.
    
    for R in range(9, 200):
        if time.time() > deadline - 5.0:
            break
        
        found_for_R = False
        for S in range(lcm_val):
            if time.time() > deadline - 5.0:
                break
            
            # Quick check: does each divisor have at least one multiple in [S, S+R]?
            candidates = []
            feasible = True
            for d in divisors:
                first_k = (S + d - 1) // d
                last_k = (S + R) // d
                if first_k > last_k:
                    feasible = False
                    break
                # Number of candidates
                count = last_k - first_k + 1
                candidates.append((first_k, count))
            
            if not feasible:
                continue
            
            # Build actual candidate values for each divisor
            cand_vals = []
            for idx, (first_k, count) in enumerate(candidates):
                d = divisors[idx]
                vals = [first_k * d + i * d for i in range(count)]
                cand_vals.append(vals)
            
            # Check if perfect matching exists using DFS with MRV
            # Order divisors by number of candidates
            order = sorted(range(n), key=lambda i: len(cand_vals[i]))
            
            used = set()
            
            def backtrack(idx):
                if idx == n:
                    return True
                i = order[idx]
                for val in cand_vals[i]:
                    if val not in used:
                        used.add(val)
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                return False
            
            if backtrack(0):
                # Reconstruct
                used2 = set()
                assignment = [None] * n
                
                def backtrack2(idx):
                    if idx == n:
                        return True
                    i = order[idx]
                    for val in cand_vals[i]:
                        if val not in used2:
                            used2.add(val)
                            assignment[i] = val
                            if backtrack2(idx + 1):
                                return True
                            used2.remove(val)
                            assignment[i] = None
                    return False
                
                if backtrack2(0):
                    result = [int(assignment[i]) for i in range(n)]
                    actual_R = max(result) - min(result)
                    if actual_R < best_R:
                        best_R = actual_R
                        best_result = result
                    found_for_R = True
                    # Don't break - keep looking for potentially better S with same R
                    # But since we're iterating R upward, the first valid R gives us
                    # a solution. Continue to find if smaller actual_R exists.
        
        if found_for_R:
            # Found a solution for this R. Since R increases, we could stop,
            # but let's check a few more R values in case a larger R window
            # happens to have a tighter actual span (unlikely but possible).
            # Actually, actual_R <= R, and we want to minimize actual_R.
            # A larger R window can't give a smaller actual_R than a smaller R window
            # that already works, because the values are still in a larger window.
            # Wait, that's not quite right. A window of size R=15 might have values
            # that span only 10, while a window of size R=12 might have values spanning 12.
            # So we should continue a bit.
            # But for simplicity, let's stop at the first R that works, since
            # the probability of finding a tighter span with larger R is low.
            break
    
    if best_result is None:
        best_result = divisors[:]
    
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from math import gcd

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)

# Precompute LCM of all divisors for potential use
_LCM_ALL = 1
for d in DIVISORS:
    _LCM_ALL = _LCM_ALL * d // gcd(_LCM_ALL, d)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    divisors = DIVISORS
    n = N
    
    # We want to find distinct multiples m_i of d_i such that max(m_i) - min(m_i) is minimized.
    # Let the window be [L, R]. We need to check if there exists a valid assignment of multiples
    # within [L, R] for all divisors, with all values distinct.
    
    # Strategy: Sliding window over L. For a given L, find the smallest possible R such that
    # a valid distinct assignment exists in [L, R].
    # To do this efficiently, we can use a greedy or backtracking approach for a fixed window [L, R].
    # However, checking every L is too slow.
    # Better: Iterate over possible window lengths W (max-min). Start small, increase W.
    # For each W, slide L from 1 to MAX_VAL-W.
    # Check if a valid assignment exists in [L, L+W].
    
    # How to check if a valid assignment exists in [L, R]?
    # This is a bipartite matching problem or a constraint satisfaction problem.
    # Since N=10 is small, we can use backtracking with pruning.
    # Order the divisors by the number of available multiples in [L, R] (most constrained first).
    
    best_width = None
    best_assignment = None
    
    # Helper: Check if a valid distinct assignment exists in [L, R]
    # Returns the assignment (list of values) if found, else None.
    def solve_window(L, R):
        # For each divisor d_i, compute the list of valid multiples in [L, R]
        # multiples_i = [m * d_i for m in range(ceil(L/d_i), floor(R/d_i)+1)]
        # We need to pick one from each list such that all are distinct.
        
        # Precompute candidate lists
        candidates = []
        for d in divisors:
            start_m = (L + d - 1) // d
            end_m = R // d
            if start_m > end_m:
                return None  # No multiple for this divisor in range
            # List of multiples
            # To speed up backtracking, sort by index? No, the values are the values.
            # We need to ensure distinctness.
            # Let's store the list of values.
            vals = list(range(start_m * d, (end_m + 1) * d, d))
            candidates.append(vals)
        
        # Sort divisors by the number of candidates (ascending) to prune early
        # We need to keep track of which divisor is which to return the correct order
        indices = list(range(n))
        indices.sort(key=lambda i: len(candidates[i]))
        
        used = set()
        assignment = [0] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            orig_idx = indices[idx]
            vals = candidates[orig_idx]
            
            for v in vals:
                if v not in used:
                    used.add(v)
                    assignment[orig_idx] = v
                    if backtrack(idx + 1):
                        return True
                    used.remove(v)
            
            return False
        
        if backtrack(0):
            return assignment
        return None

    # Optimization: The window size W doesn't need to be huge.
    # The density of multiples is high.
    # Let's estimate a good starting W.
    # The "average" gap between multiples of different divisors is small.
    # Let's try W from 10 upwards? Or start with a larger guess and refine?
    # Actually, let's just iterate W from a small number to a larger number.
    # But sliding L for each W is O(MAX_VAL * N * Backtrack).
    # MAX_VAL is 1e6. This might be too slow if we check many W's.
    
    # Alternative: Iterate L. For each L, find the minimal R.
    # But finding minimal R for a fixed L is also a search.
    
    # Let's combine:
    # Iterate W from 1 to some limit (e.g., 100).
    # For each W, iterate L from 1 to MAX_VAL - W.
    # This is 100 * 1e6 = 1e8 checks. Too slow.
    
    # We need a smarter approach.
    # Notice that the pattern of available multiples repeats with period LCM(divisors).
    # LCM(6,10,14,15,21,22,26,33,35,39) is large.
    # Let's calculate it.
    # 6=2*3, 10=2*5, 14=2*7, 15=3*5, 21=3*7, 22=2*11, 26=2*13, 33=3*11, 35=5*7, 39=3*13
    # Primes: 2,3,5,7,11,13
    # LCM = 2 * 3 * 5 * 7 * 11 * 13 = 30030.
    # So the pattern of "which divisors have a multiple at position x" repeats every 30030.
    # However, the specific VALUES of the multiples depend on L.
    # But the RELATIVE positions of the multiples within the window [L, L+W] depend on L mod 30030.
    # So we only need to check L in [1, 30030].
    
    LCM_PERIOD = 30030
    
    # Let's verify LCM calculation
    l = 1
    for d in divisors:
        l = l * d // gcd(l, d)
    # assert l == 30030, f"LCM is {l}"
    
    # So we iterate W from 1 to, say, 100.
    # For each W, we iterate L from 1 to LCM_PERIOD.
    # If a solution is found for L, we can use it (as long as L+W <= MAX_VAL, which is true for small W).
    # Wait, the multiples must be <= MAX_VAL.
    # If L is small, the multiples are small.
    # If we find a valid window [L, L+W], the values are in [L, L+W].
    # We just need to ensure L+W <= MAX_VAL.
    # Since W is small (< 1000) and L < 30030, L+W < 31000 < 1e6. So this is always valid.
    
    # Let's try this.
    # For W in range(1, 200):
    #   for L in range(1, LCM_PERIOD + 1):
    #     if time.time() > deadline: return best or divisors
    #     if solve_window(L, L+W):
    #       return assignment
    
    # This is 200 * 30030 = 6,006,000 calls to solve_window.
    # solve_window is a backtracking search.
    # In the worst case, it might be slow. But for small W, the candidate lists are short.
    # For W=1, most divisors have 0 or 1 candidate.
    # For W=10, candidates are ~1-2 per divisor.
    # Backtracking should be very fast.
    
    # Let's add a heuristic: sort divisors by candidate count in solve_window.
    # We already do that.
    
    # One more optimization:
    # If W < max(divisors), it's impossible? No.
    # Example: d=6, d=10. W=4.
    # L=1, R=5.
    # 6: no multiple.
    # So we need L such that all divisors have at least one multiple in [L, L+W].
    # This requires W >= max(d_i) - min(d_i)? No.
    # It requires that the gap between consecutive "coverage" is small.
    
    # Let's just run the search.
    
    # To speed up solve_window, we can pre-check:
    # For each divisor, is there at least one multiple in [L, R]?
    # If not, return None immediately.
    # This is already done in the candidate generation (if start_m > end_m).
    
    # Let's refine the loop bounds.
    # We don't need to check all L from 1 to LCM_PERIOD for every W.
    # We can just iterate L from 1 to LCM_PERIOD, and for each L, find the minimal W?
    # No, finding minimal W for a fixed L is harder.
    # The W-outer loop is better because we can stop as soon as we find ANY valid window.
    
    # Let's start W from 1.
    # Actually, W must be at least max(d_i) - min(d_i) ? No.
    # But W must be at least the difference between the smallest and largest multiple we pick.
    # The smallest possible multiple for d_i is d_i (if L=1).
    # So min possible value is min(d_i) = 6.
    # Max possible value is max(d_i) = 39.
    # So W >= 39 - 6 = 33?
    # Not necessarily, because we can shift L.
    # But the "spread" of the divisors themselves is 33.
    # So W must be at least 33?
    # Let's check.
    # If we pick m_i = d_i, then values are 6,10,14,15,21,22,26,33,35,39.
    # Max=39, Min=6, W=33.
    # Can we do better?
    # We need to shift some multiples to be closer.
    # E.g., for d=6, pick 12, 18, 24...
    # For d=10, pick 10, 20, 30...
    # If we pick 12 (for 6) and 10 (for 10), they are close.
    # Let's try to find a cluster.
    # The theoretical minimum W is likely small, maybe around 10-20.
    
    # So iterating W from 1 to 50 should be enough.
    # 50 * 30030 = 1.5 million calls.
    # This is very fast.
    
    for W in range(1, 100):
        if time.time() > deadline:
            break
        for L in range(1, LCM_PERIOD + 1):
            if time.time() > deadline:
                break
            R = L + W
            if R > MAX_VAL:
                break
            res = solve_window(L, R)
            if res is not None:
                # Found a valid assignment
                # Ensure types are int
                return [int(x) for x in res]
        
        if time.time() > deadline:
            break
    
    # Fallback: return the divisors themselves (valid, distinct, W=33)
    return [int(d) for d in divisors]

# EVOLVE-BLOCK-END

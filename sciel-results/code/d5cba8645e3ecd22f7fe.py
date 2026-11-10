"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from math import gcd
from functools import lru_cache

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
LCM = 30030  # LCM of all divisors

def _get_multiples_in_window(d, lo, hi):
    """Get all multiples of d in [lo, hi]"""
    if hi < lo:
        return []
    start_k = (lo + d - 1) // d
    end_k = hi // d
    if start_k > end_k:
        return []
    return list(range(start_k * d, end_k * d + 1, d))

def _can_assign(window_lo, window_hi):
    """
    Check if we can assign one distinct multiple of each divisor within [window_lo, window_hi].
    Uses backtracking with constraint propagation.
    """
    n = len(DIVISORS)
    # For each divisor, get candidate multiples in the window
    candidates = []
    for d in DIVISORS:
        cands = _get_multiples_in_window(d, window_lo, window_hi)
        if not cands:
            return None
        candidates.append(cands)
    
    # Sort divisors by number of candidates (most constrained first)
    order = sorted(range(n), key=lambda i: len(candidates[i]))
    
    used = set()
    result = [None] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        i = order[idx]
        for val in candidates[i]:
            if val not in used:
                used.add(val)
                result[i] = val
                if backtrack(idx + 1):
                    return True
                used.remove(val)
                result[i] = None
        return False
    
    if backtrack(0):
        return result
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # First, try the obvious solution: smallest multiples
    # 6,10,14,15,21,22,26,33,35,39 -> range = 33
    base_solution = DIVISORS[:]
    base_range = max(base_solution) - min(base_solution)
    
    # Try to find a better solution by searching windows
    # The minimum possible range for 10 distinct integers is 9
    # Let's search for ranges from 9 upward
    
    best_solution = base_solution
    best_range = base_range
    
    # Binary search on the window size
    # Lower bound: 9 (10 distinct integers)
    # Upper bound: base_range (33)
    
    lo_range = 9
    hi_range = best_range
    
    # First check if any range smaller than base_range works
    # We'll try ranges from 9 to base_range-1
    
    # For efficiency, we can try to find the optimal by checking
    # if a window of size W can accommodate all divisors
    
    # Strategy: try each possible range from 9 upward
    # For each range W, try different window positions
    
    # Since LCM is 30030, the pattern repeats. We only need to check
    # window positions modulo 30030, but shifted appropriately.
    
    # Actually, let's be smarter. For a given range W, we need to find
    # a window [L, L+W] such that for each divisor d, there's at least
    # one multiple of d in [L, L+W], and we can assign distinct values.
    
    # Key insight: for small W, the number of multiples of each d in the
    # window is small, making backtracking fast.
    
    # Let's try ranges from 9 to best_range-1
    for W in range(9, best_range):
        if time.time() > deadline:
            break
        
        # For this window size W, try different starting positions
        # The starting position L determines which multiples are available.
        # Due to periodicity, we only need to check L mod LCM, but we need
        # to be careful about the window boundaries.
        
        # For a given W, the critical values of L are those where the
        # window boundaries align with multiples of the divisors.
        
        # Let's try a set of candidate L values:
        # - L = 0, 1, 2, ... up to some limit
        # - But we can be smarter: only L values that are near multiples
        
        # Actually, let's just try a range of L values. Since the pattern
        # repeats every LCM=30030, and W is at most 32, we only need to
        # check L in a range of about LCM+W.
        
        # But that's too many. Let's think differently.
        
        # For each divisor d, the multiples in [L, L+W] depend on L mod d.
        # The "critical" L values are those where L mod d changes which
        # multiples are in the window.
        
        # Let's try L values that are multiples of small numbers minus small offsets.
        
        # Simpler approach: try L from 0 to some reasonable limit
        # Since we want small values, let's try L from 0 to 200
        
        found = False
        for L in range(0, 200):
            if time.time() > deadline:
                break
            
            window_lo = L
            window_hi = L + W
            
            # Quick check: for each divisor, is there at least one multiple in the window?
            feasible = True
            for d in DIVISORS:
                if not _get_multiples_in_window(d, window_lo, window_hi):
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            # Try to assign
            result = _can_assign(window_lo, window_hi)
            if result is not None:
                # Found a solution with range W
                # Verify all values are positive and <= MAX_VAL
                if all(v > 0 and v <= MAX_VAL for v in result) and len(set(result)) == 10:
                    best_solution = result
                    best_range = W
                    found = True
                    break
        
        if found:
            # We found a solution with this range, can we do better?
            # Continue searching for smaller ranges
            continue
    
    # Ensure we return Python ints in a list
    return [int(v) for v in best_solution]

# EVOLVE-BLOCK-END

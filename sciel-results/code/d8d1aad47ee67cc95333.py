"""
Combines the structural search loop of the current program with the 
backtracking assignment logic from Inspiration 1 to handle distinctness 
constraints more effectively than the greedy 'smallest multiple' approach.
"""
import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
_DIVS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_MAXV = 1000000
_DEADLINE = 80.0  # wall-clock seconds (90s kill, leave margin)

def _lcm(a, b):
    return a // gcd(a, b) * b

# Precompute LCM of all divisors (not strictly needed for the scan, but good for context)
_LCM_ALL = reduce(_lcm, _DIVS)

def _try_assign(L, W):
    """
    Try to assign distinct multiples of each divisor within [L, L+W].
    Returns list of assigned values (aligned with _DIVS order) or None.
    Uses backtracking with MRV (Minimum Remaining Values) heuristic.
    """
    # Precompute candidate multiples for each divisor
    candidates = []
    for d in _DIVS:
        cands = []
        # First multiple >= L
        k0 = (L + d - 1) // d
        # Last multiple <= L+W
        k1 = (L + W) // d
        if k0 <= k1:
            # Generate candidates
            # If the window is large, this list could be big, but W is expected to be small
            cands = [k * d for k in range(k0, k1 + 1)]
        if not cands:
            return None
        candidates.append(cands)
    
    # Order divisors by number of candidates (MRV)
    # We need to map back to original indices
    indices = list(range(len(_DIVS)))
    indices.sort(key=lambda i: len(candidates[i]))
    
    assigned = [None] * len(_DIVS)
    used = set()
    
    def backtrack(idx):
        if idx == len(indices):
            return True
        i = indices[idx]
        for val in candidates[i]:
            if val not in used:
                used.add(val)
                assigned[i] = val
                if backtrack(idx + 1):
                    return True
                used.discard(val)
                assigned[i] = None
        return False
    
    if backtrack(0):
        return assigned
    return None

def construct():
    t0 = time.time()
    
    # Lower bound on W: 10 distinct integers in a closed interval of length W
    # implies W >= 9.
    # However, due to divisibility constraints, W will likely be larger.
    # We start from 9 and go up.
    
    # We scan L from 1 to a limit. 
    # If a solution exists for a given W, it likely exists for small L 
    # due to the periodicity and density of multiples.
    # We use a limit of 50,000 for L. This is fast enough to be safe.
    L_LIMIT = 50000
    
    for W in range(9, 200):
        if time.time() - t0 > _DEADLINE:
            break
        
        # Scan L
        # We can optimize by stepping L, but 1 is safest for correctness.
        # Given backtracking is fast, 50k iterations * 10 divisors is ~500k ops, 
        # which is < 0.1s in Python if optimized, but backtracking overhead exists.
        # Let's try 10,000 first, then expand if needed? 
        # Actually, let's just scan 1..10000. If not found, try next W.
        # If we miss a solution at L=20000, we'll find it at a larger W, 
        # but that's suboptimal. 
        # However, the probability of a solution existing for small W but 
        # only at large L is low.
        
        found = False
        for L in range(1, L_LIMIT + 1):
            if time.time() - t0 > _DEADLINE:
                break
            if L + W > _MAXV:
                break
            reps = _try_assign(L, W)
            if reps is not None:
                # Found optimal W and a valid assignment
                return [int(x) for x in reps]
        
        # If we finished scanning L for this W without success, try next W
    
    # Fallback: Random search or return divisors
    # If we didn't find a small W, we might have timed out or missed a solution.
    # As a last resort, return the divisors themselves (valid, but large spread).
    # Or try a random search for a better solution in remaining time.
    import random
    random.seed(42)
    best_reps = None
    best_W = float('inf')
    
    # Try random L for a few W values if we have time
    for W in range(9, 100):
        if time.time() - t0 > _DEADLINE - 5:
            break
        for _ in range(1000):
            if time.time() - t0 > _DEADLINE - 5:
                break
            L = random.randint(1, _MAXV - W)
            reps = _try_assign(L, W)
            if reps is not None:
                if W < best_W:
                    best_reps = reps
                    best_W = W
                    # We can't return immediately here because we might find 
                    # a smaller W in a later random iteration? 
                    # No, we are iterating W in increasing order. 
                    # But random search doesn't guarantee finding the *first* valid L.
                    # So we just keep the best found so far.
    
    if best_reps is not None:
        return [int(x) for x in best_reps]
    
    # Ultimate fallback
    return [int(d) for d in _DIVS]
# EVOLVE-BLOCK-END

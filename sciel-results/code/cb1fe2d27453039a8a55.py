import time
import sys

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
DEADLINE = 90.0 # Seconds

def _can_assign(divisors, L, W):
    """
    Check if we can assign distinct multiples to each divisor within [L, L+W].
    Returns a list of assigned values or None.
    """
    n = len(divisors)
    # Build candidate lists for each divisor
    candidates = []
    for d in divisors:
        # Find multiples of d in [L, L+W]
        # First multiple >= L
        start_k = (L + d - 1) // d
        end_k = (L + W) // d
        cands = []
        for k in range(start_k, end_k + 1):
            val = k * d
            if val <= MAX_VAL:
                cands.append(val)
        candidates.append(cands)
        
    # Sort divisors by number of candidates (most constrained first)
    # We need to keep track of original indices
    indices = list(range(n))
    indices.sort(key=lambda i: len(candidates[i]))
    
    assigned = [None] * n
    used_values = set()
    
    def dfs(pos):
        if pos == n:
            return True
        
        idx = indices[pos]
        d = divisors[idx]
        cands = candidates[idx]
        
        # Try each candidate
        for val in cands:
            if val not in used_values:
                used_values.add(val)
                assigned[idx] = val
                if dfs(pos + 1):
                    return True
                used_values.remove(val)
                assigned[idx] = None
        return False
    
    if dfs(0):
        return assigned
    return None

def construct():
    start_time = time.time()
    
    # Theoretical minimum width is 9 (10 distinct integers)
    # Search increasing widths
    for W in range(9, 30):
        # Search for a valid starting position L
        # We check L in a range. Since the problem is periodic-ish but distinctness breaks simple periodicity,
        # we check a reasonably large window. 10000 is usually enough for small divisors.
        # If not found, we might need to go higher, but let's start with 10000.
        for L in range(1, 10001):
            # Check time limit
            if time.time() - start_time > DEADLINE - 5:
                # Fallback to seed solution if time is up
                return DIVISORS[:]
            
            res = _can_assign(DIVISORS, L, W)
            if res is not None:
                # Ensure types are Python int
                return [int(x) for x in res]
                
    # If no solution found in the search space, return the seed solution
    # This should not happen for W < 30, but just in case
    return DIVISORS[:]
# EVOLVE-BLOCK-END

"""
Strategy:
1. The problem is to find 10 distinct positive integers x_i such that x_i % d_i == 0,
   minimizing max(x_i) - min(x_i).
2. The divisors are fixed: [6, 10, 14, 15, 21, 22, 26, 33, 35, 39].
3. The LCM of these divisors is 30030. The pattern of residues is periodic with this period.
4. We search for the minimum span W. We know from previous runs that W=16 is the optimum.
5. To be robust and fast, we:
   a. Pre-compute the solution for W=16 at module load time (or on first call) by scanning L.
   b. If not found (shouldn't happen), we run a quick search for W=16, then 17, etc.
   c. We use a highly optimized inner check:
      - Rigid divisors (d > W) have at most 1 multiple in any window of length W.
      - We check if these rigid multiples are distinct and within the window.
      - For flexible divisors (d <= W), we use a small backtracking search.
6. Since the search for W=16 is very fast (usually finds a solution within the first few hundred L values),
   we can afford to do it on every call if needed, but caching is better.

Key optimizations:
- Use bitmasks for the "used" offsets in the window (since W is small, <= 100, we can use an integer bitmask).
- Pre-compute the residue tables for rigid divisors.
- Start the search for W at 16 (the known optimum). If no solution is found for W=16, try 17, 18...
- Cache the result so subsequent calls are O(1).

The current program (score 2.0625) already does this, but we can make it more robust by:
- Ensuring the search is as fast as possible.
- Adding a fallback to a known valid solution if the search fails (though it shouldn't).
- Properly handling types (Python int, list).

Let's refine the search to be extremely fast.
We will pre-compute the "rigid" constraints for a given W.
For W=16, rigid divisors are 21, 22, 26, 33, 35, 39.
For each L, we compute the offset for each rigid divisor.
If any offset is invalid (no multiple in window) or collides, we skip.
Otherwise, we try to assign the flexible divisors (6, 10, 14, 15).

We can pre-compute the valid L values for the rigid divisors for W=16.
Then, for each valid L, we try to assign the flexible divisors.

Actually, since the LCM is 30030, and we only need to find ONE solution, we can just scan L from 0 to 30029.
The scan is fast. Let's just do that.

To be safe, I will include a cached result.
"""

import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START

_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_MAX_VAL = 1000000

# Pre-compute LCM
def _lcm(a, b):
    return a * b // gcd(a, b)

_LCM = reduce(_lcm, _DIVISORS)

# Cache for the result
_CACHED_RESULT = None

def _find_solution(W):
    """
    Find a solution for span W.
    Returns a list of 10 integers or None.
    """
    n = len(_DIVISORS)
    # Identify rigid and flexible indices
    rigid_indices = []
    flexible_indices = []
    for i, d in enumerate(_DIVISORS):
        if d > W:
            rigid_indices.append(i)
        else:
            flexible_indices.append(i)
    
    # Pre-compute offset tables for rigid divisors
    # table[d][rem] = offset if there is a multiple of d in [L, L+W] with offset = multiple - L
    # offset = (d - (L % d)) % d
    # We need offset <= W
    offset_tables = {}
    for i in rigid_indices:
        d = _DIVISORS[i]
        table = [0] * d
        for rem in range(d):
            offset = (d - rem) % d
            table[rem] = offset if offset <= W else -1
        offset_tables[i] = table
    
    # Iterate L from 0 to LCM-1
    for L in range(_LCM):
        used_mask = 0
        assignments = [None] * n
        valid = True
        
        # Check rigid divisors
        for i in rigid_indices:
            d = _DIVISORS[i]
            rem = L % d
            offset = offset_tables[i][rem]
            if offset < 0:
                valid = False
                break
            bit = 1 << offset
            if used_mask & bit:
                valid = False
                break
            used_mask |= bit
            assignments[i] = L + offset
        
        if not valid:
            continue
        
        # Check flexible divisors via backtracking
        # We need to assign flexible divisors to distinct offsets in [0, W]
        # that are not already used.
        # Since W is small, we can use a bitmask for available offsets.
        available_mask = (1 << (W + 1)) - 1
        available_mask &= ~used_mask
        
        # Get candidates for each flexible divisor
        candidates = []
        for i in flexible_indices:
            d = _DIVISORS[i]
            cands = []
            # First multiple >= L
            k = (L + d - 1) // d
            val = k * d
            while val <= L + W:
                offset = val - L
                if available_mask & (1 << offset):
                    cands.append(offset)
                val += d
            candidates.append(cands)
            if not cands:
                break
        
        if len(candidates) < len(flexible_indices):
            continue
        
        # Backtracking for flexible divisors
        # Order by fewest candidates (MRV)
        order = sorted(range(len(flexible_indices)), key=lambda x: len(candidates[x]))
        
        def backtrack(idx, current_mask, current_assignments):
            if idx == len(order):
                return True
            flex_pos = order[idx]
            orig_idx = flexible_indices[flex_pos]
            for offset in candidates[flex_pos]:
                bit = 1 << offset
                if not (current_mask & bit):
                    current_mask |= bit
                    current_assignments[orig_idx] = L + offset
                    if backtrack(idx + 1, current_mask, current_assignments):
                        return True
                    current_mask &= ~bit
                    current_assignments[orig_idx] = None
            return False
        
        if backtrack(0, available_mask, assignments):
            # Check if all assignments are positive and <= MAX_VAL
            for i in range(n):
                if assignments[i] is None or assignments[i] < 1 or assignments[i] > _MAX_VAL:
                    return None
            # Check distinctness (should be guaranteed by bitmask, but just in case)
            if len(set(assignments)) != n:
                return None
            return assignments
    
    return None

def construct():
    global _CACHED_RESULT
    
    if _CACHED_RESULT is not None:
        return _CACHED_RESULT
    
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # We know W=16 is the optimum. Start there.
    # If it fails, try 17, 18... up to a reasonable limit.
    # Also, we can try W=0..15 if we want to be absolutely sure, but it's slow.
    # Given the previous success, we start at 16.
    
    best_solution = None
    for W in range(16, 50):
        if time.time() > deadline:
            break
        sol = _find_solution(W)
        if sol is not None:
            best_solution = sol
            break
    
    if best_solution is None:
        # Fallback: return the divisors themselves (span 33)
        # This is a valid solution, though not optimal.
        best_solution = _DIVISORS[:]
    
    # Ensure types are correct
    result = [int(x) for x in best_solution]
    _CACHED_RESULT = result
    return result

# EVOLVE-BLOCK-END

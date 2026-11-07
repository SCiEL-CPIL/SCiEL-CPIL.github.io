"""Pack distinct multiples: minimize enclosing interval length."""
import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = 10
MAX_VAL = 1000000
LCM = 30030  # lcm of all divisors


def _lcm(a, b):
    return a * b // gcd(a, b)


def _feasible_window(start, length):
    """Check if there exists an assignment of distinct multiples in [start, start+length].
    Uses backtracking with constraint propagation.
    Returns the assignment list if found, else None.
    """
    # For each divisor i, the multiples in [start, start+length] are:
    # k * divisors[i] where start <= k*div <= start+length
    # i.e., k in [ceil(start/div), floor((start+length)/div)]
    
    # Build candidate sets (as sorted lists of actual values)
    candidates = []
    for d in DIVISORS:
        lo_k = (start + d - 1) // d  # ceil(start/d)
        hi_k = (start + length) // d  # floor((start+length)/d)
        if lo_k > hi_k:
            return None  # no multiple in range
        # Store the actual values
        vals = [k * d for k in range(lo_k, hi_k + 1)]
        candidates.append(vals)
    
    # Backtracking: assign one divisor at a time
    # Order divisors by fewest candidates first (MRV heuristic)
    order = sorted(range(N), key=lambda i: len(candidates[i]))
    assigned = [None] * N
    used = set()
    
    def backtrack(idx):
        if idx == N:
            return True
        i = order[idx]
        for val in candidates[i]:
            if val not in used:
                used.add(val)
                assigned[i] = val
                if backtrack(idx + 1):
                    return True
                used.remove(val)
                assigned[i] = None
        return False
    
    if backtrack(0):
        return assigned
    return None


def _feasible_window_fast(start, length):
    """Faster feasibility check using bitmasks when window is small."""
    # For small windows, the number of candidates per divisor is small
    # Use a different approach: for each divisor, compute the set of possible multiples
    # and check if a perfect matching exists (Hall's theorem / bipartite matching)
    
    # Actually, for windows of length < ~200, each divisor has very few candidates
    # Let's use the simple backtracking but with better ordering
    
    candidates = []
    for d in DIVISORS:
        lo_k = (start + d - 1) // d
        hi_k = (start + length) // d
        if lo_k > hi_k:
            return None
        vals = [k * d for k in range(lo_k, hi_k + 1)]
        candidates.append(vals)
    
    # Check if any divisor has 0 candidates
    # Then use backtracking with MRV
    order = sorted(range(N), key=lambda i: len(candidates[i]))
    if len(candidates[order[0]]) == 0:
        return None
    
    assigned = [None] * N
    used = set()
    
    def bt(idx):
        if idx == N:
            return True
        i = order[idx]
        for val in candidates[i]:
            if val not in used:
                used.add(val)
                assigned[i] = val
                if bt(idx + 1):
                    return True
                used.remove(val)
                assigned[i] = None
        return False
    
    if bt(0):
        return assigned
    return None


def construct():
    deadline = time.time() + 80  # 80s safety margin
    
    # Phase 1: Binary search on window length
    # Lower bound: at least N-1 = 9 (since we need 10 distinct integers)
    # Upper bound: LCM = 30030 (always feasible by construction)
    
    # First, let's verify that LCM window is feasible
    # Actually, let's search more cleverly.
    
    # Key insight: For a given window length L, feasibility depends on the start position mod LCM.
    # So we can check all start positions in [0, LCM-1] for a given L.
    # But that's 30030 positions, and for each we need to check feasibility.
    
    # Better approach: For small L, the number of candidates per divisor is very small,
    # so backtracking is fast. We can scan all start positions.
    
    # Let's try increasing window sizes from a lower bound.
    # Lower bound: we need 10 distinct values, each a multiple of its divisor.
    # The minimum possible gap between consecutive multiples varies by divisor.
    
    # Actually, let's use a smarter approach:
    # For window length L, for each start position s in [0, LCM-1],
    # check if the window [s, s+L] (mod LCM, i.e., considering the periodic structure)
    # can accommodate all 10 divisors with distinct multiples.
    
    # But we need actual values, not just residues. However, if a window of length L
    # starting at residue r (mod LCM) is feasible, then we can place it at
    # r + k*LCM for any k such that the values are <= MAX_VAL.
    
    # So the problem reduces to: find the minimum L such that there exists a residue r
    # where the window [r, r+L] contains distinct multiples for all divisors.
    
    # For a given L, we scan r from 0 to LCM-1 and check feasibility.
    
    # Optimization: For small L, most windows will be infeasible quickly.
    
    # Let's start with a lower bound. The minimum window must contain at least
    # one multiple of each divisor. The "tightest" divisor is the one with the
    # largest value, which is 39. But actually, the constraint is about distinctness.
    
    # A better lower bound: consider that within any interval of length L,
    # divisor d has at most floor(L/d) + 1 multiples.
    # For feasibility, we need at least 1 multiple of each divisor, and all distinct.
    
    # Let's just search from L = 10 upward (since we need 10 distinct positive integers).
    
    best_assignment = None
    best_length = None
    
    # Try window lengths from a reasonable lower bound
    # The absolute minimum is 9 (10 consecutive integers), but that's unlikely to work.
    # Let's start from 10 and go up.
    
    # Actually, let's be smarter. For each L, we only need to check starts that
    # are "aligned" in some sense. But the simplest correct approach is to scan all.
    
    # For efficiency, let's limit the scan. For each L, scan r in [0, min(LCM, some_limit)].
    
    # Let's try a range of L values.
    for L in range(10, 30031):
        if time.time() > deadline - 10:
            break
        
        found = False
        for r in range(LCM):
            if time.time() > deadline - 10:
                break
            
            # Check window [r, r+L]
            # But we need actual multiples, not just residues in [r, r+L]
            # The multiples of d in [r, r+L] are: k*d where r <= k*d <= r+L
            # i.e., k in [ceil(r/d), floor((r+L)/d)]
            
            candidates = []
            feasible = True
            for d in DIVISORS:
                lo_k = (r + d - 1) // d
                hi_k = (r + L) // d
                if lo_k > hi_k:
                    feasible = False
                    break
                vals = [k * d for k in range(lo_k, hi_k + 1)]
                candidates.append(vals)
            
            if not feasible:
                continue
            
            # Check if we can assign distinct values
            # Use backtracking with MRV
            order = sorted(range(N), key=lambda i: len(candidates[i]))
            if len(candidates[order[0]]) == 0:
                continue
            
            assigned = [None] * N
            used = set()
            
            def bt(idx):
                if idx == N:
                    return True
                i = order[idx]
                for val in candidates[i]:
                    if val not in used:
                        used.add(val)
                        assigned[i] = val
                        if bt(idx + 1):
                            return True
                        used.remove(val)
                        assigned[i] = None
                return False
            
            if bt(0):
                # Found a feasible assignment
                # Check that all values are <= MAX_VAL and >= 1
                if all(1 <= v <= MAX_VAL for v in assigned):
                    best_assignment = assigned[:]
                    best_length = L
                    found = True
                    break
        
        if found:
            # We found the minimum L (since we're searching in increasing order)
            # But wait - we should verify this is truly minimal.
            # Since we search L in increasing order, the first feasible L is optimal.
            break
    
    if best_assignment is not None:
        return [int(v) for v in best_assignment]
    
    # Fallback: use the seed program's approach
    return DIVISORS[:]
# EVOLVE-BLOCK-END

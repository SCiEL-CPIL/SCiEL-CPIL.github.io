"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

_DIVS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_N = len(_DIVS)
_MAX = 1000000

def _lcm(a, b):
    return a // gcd(a, b) * b

_LCM = 1
for _d in _DIVS:
    _LCM = _lcm(_LCM, _d)

def _feasible(L, R):
    """Check if there exists a distinct assignment of multiples within [L, L+R].
    Uses bipartite matching (Hopcroft-Karp style, but n=10 so simple DFS suffices)."""
    # For each divisor i, find which values in [L, L+R] are multiples of _DIVS[i]
    # We need a perfect matching in the bipartite graph.
    # n=10, so we can use bitmask DP or simple augmenting path.
    
    # Build adjacency: for each divisor i, list of candidate positions (0..R)
    # where L + pos is divisible by _DIVS[i]
    adj = []
    for i in range(_N):
        d = _DIVS[i]
        # Find first multiple of d >= L
        first = ((L + d - 1) // d) * d
        if first > L + R:
            return None  # no candidate for this divisor
        # Generate all multiples in [L, L+R]
        cands = []
        v = first
        while v <= L + R:
            cands.append(v - L)  # position in window
            v += d
        adj.append(cands)
    
    # Check Hall's condition quickly: for each divisor, must have at least 1 candidate
    for cands in adj:
        if not cands:
            return None
    
    # Use bitmask DP: dp[mask] = set of positions used, or more efficiently:
    # dp[i][mask] = True if we can assign first i divisors to positions in mask
    # Since R can be large, use matching instead.
    
    # Simple augmenting path matching (n=10 left nodes, up to R+1 right nodes)
    # But R could be large. Let's use a different approach:
    # Since n=10 is small, use recursive backtracking with memoization on (i, used_mask_of_positions)
    # But positions can be up to R which might be large...
    
    # Alternative: since we just need existence, use the fact that n is tiny.
    # Try all permutations? 10! = 3.6M, too slow for many calls.
    
    # Better: use bipartite matching with Hopcroft-Karp.
    # Left side: 0..9 (divisors), Right side: 0..R (positions)
    # But R might be large. However, each divisor only connects to multiples,
    # so the right side is sparse.
    
    # Actually, let's just do DFS matching. n=10, so it's fast.
    match_r = {}  # position -> divisor index
    
    def try_assign(i, visited):
        for pos in adj[i]:
            if pos in visited:
                continue
            visited.add(pos)
            if pos not in match_r or try_assign(match_r[pos], visited):
                match_r[pos] = i
                return True
        return False
    
    result = 0
    for i in range(_N):
        visited = set()
        if try_assign(i, visited):
            result += 1
        else:
            return None  # no perfect matching
    
    if result < _N:
        return None
    
    # Extract assignment
    assignment = [0] * _N
    for pos, i in match_r.items():
        assignment[i] = L + pos
    
    # Verify distinctness
    if len(set(assignment)) != _N:
        return None
    
    return assignment

def construct():
    start = time.time()
    deadline = start + 85.0  # 85s deadline, 5s safety margin
    
    best = None
    best_R = float('inf')
    
    # Theoretical minimum R: we need 10 distinct integers, so R >= 9
    # Search over R from 9 upward, and for each R, scan L over one LCM period
    # L ranges from 0 to LCM-1 (since the pattern repeats every LCM)
    # But we also need L + R <= _MAX, so L <= _MAX - R
    
    # For efficiency, scan R from 9 upward, and for each R try all L in [0, min(LCM-1, _MAX-R)]
    # But LCM=30030, so that's up to 30030 L values per R.
    # We can early-stop once we find a valid assignment for a given R.
    
    # Optimization: sort divisors by number of multiples in a typical window
    # to help matching converge faster.
    
    for R in range(9, 200):  # upper bound on R to search
        if time.time() > deadline:
            break
        max_L = min(_LCM - 1, _MAX - R)
        if max_L < 0:
            break
        
        found = False
        for L in range(0, max_L + 1):
            if time.time() > deadline:
                break
            assign = _feasible(L, R)
            if assign is not None:
                # Verify all constraints
                valid = True
                for i, v in enumerate(assign):
                    if v <= 0 or v > _MAX or v % _DIVS[i] != 0:
                        valid = False
                        break
                if valid and len(set(assign)) == _N:
                    if best is None or R < best_R:
                        best = assign
                        best_R = R
                        found = True
                        break  # found for this R, move to next R
        
        if found:
            # Found a valid assignment with this R; try smaller R? 
            # No, we're going from 9 upward, so first found is optimal.
            # But we should verify it's truly minimal. Since we start at 9,
            # the first R that works is the minimum.
            break
    
    if best is None:
        # Fallback: use the trivial solution
        return _DIVS[:]
    
    # Ensure Python int types
    return [int(v) for v in best]
# EVOLVE-BLOCK-END

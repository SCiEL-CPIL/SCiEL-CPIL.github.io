"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

N = 12
LCM_12 = 27720  # LCM of 1..12

def can_assign(m, L):
    """Check if we can assign distinct b_i in (m, m+L] with i | b_i for i=1..12.
    Uses bitmask for used values for speed."""
    # Map values to indices 0..L-1 corresponding to m+1..m+L
    # We only care about values in (m, m+L]
    max_val = m + L
    min_val = m + 1
    
    candidates = []
    for i in range(1, N + 1):
        first = ((m // i) + 1) * i
        if first > max_val:
            return False
        cands = []
        val = first
        while val <= max_val:
            # Index in the window: val - min_val
            idx = val - min_val
            cands.append(idx)
            val += i
        if not cands:
            return False
        candidates.append(cands)
    
    # Order by most constrained (fewest candidates) first
    order = sorted(range(N), key=lambda i: len(candidates[i]))
    
    def backtrack(pos, used_mask):
        if pos == N:
            return True
        i = order[pos]
        for idx in candidates[i]:
            bit = 1 << idx
            if not (used_mask & bit):
                if backtrack(pos + 1, used_mask | bit):
                    return True
        return False
    
    return backtrack(0, 0)

def compute_f(m):
    """Compute f(12, m): least positive L such that (m, m+L] has distinct valid b_i."""
    lo, hi = 1, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if can_assign(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    best_m = 0
    best_L = -1
    for m in range(LCM_12):
        L = compute_f(m)
        if L > best_L:
            best_L = L
            best_m = m
    return best_m
# EVOLVE-BLOCK-END

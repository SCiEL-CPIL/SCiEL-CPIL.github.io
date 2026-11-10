"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

N = 12
LCM_12 = 27720  # LCM of 1..12

def can_assign(m, L):
    """Check if we can assign distinct b_i in (m, m+L] with i | b_i for i=1..12."""
    cands = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        r = (-m) % i
        start = r if r != 0 else i
        if start > L:
            return False
        cands[i] = list(range(start, L + 1, i))
    
    # Order by most constrained (fewest candidates) first
    order = sorted(range(1, N + 1), key=lambda i: len(cands[i]))
    used = set()
    
    def backtrack(pos):
        if pos == N:
            return True
        i = order[pos]
        for k in cands[i]:
            if k not in used:
                used.add(k)
                if backtrack(pos + 1):
                    return True
                used.remove(k)
        return False
    
    return backtrack(0)

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
            if best_L == N:
                return best_m
    return best_m
# EVOLVE-BLOCK-END

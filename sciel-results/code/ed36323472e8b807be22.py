import time
import math

def _has_perfect_matching(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1,...,b_n with i | b_i."""
    adj = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, L + 1):
            if (m + j) % i == 0:
                mask |= (1 << (j - 1))
        adj[i] = mask
        if mask == 0:
            return False
    
    match_r = [0] * L
    
    def dfs(i, visited):
        candidates = adj[i] & ~visited
        while candidates:
            low_bit = candidates & -candidates
            j_idx = low_bit.bit_length() - 1
            candidates ^= low_bit
            if match_r[j_idx] == 0 or dfs(match_r[j_idx], visited | low_bit):
                match_r[j_idx] = i
                return True
        return False
    
    for i in range(1, n + 1):
        if not dfs(i, 0):
            return False
    return True

def _compute_f(m, n=12, max_L=144):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    lo, hi = 1, max_L
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 12
    max_m = 1000000
    P = 27720  # LCM(1..12)
    
    best_m = 0
    best_f = 0
    
    # Key insight: f(12, m) depends only on m mod P.
    # Precompute f for all residues 0..P-1, find the best one.
    # Then pick the smallest m <= max_m with that residue.
    
    # We can speed up by noting that for a given residue r = m mod P,
    # the divisibility pattern (m+j) % i == 0 depends only on (r+j) % i.
    # So we compute using representative m = r (which is in 0..P-1).
    
    residues_to_check = list(range(P))
    
    # Sort by potential: residues where many i have few candidates in small windows
    # For now, just scan all residues with a time guard
    
    for r in residues_to_check:
        if time.time() > deadline:
            break
        f_val = _compute_f(r, n)
        if f_val > best_f:
            best_f = f_val
            best_m = r
    
    # Now find the smallest m in [0, max_m] with m % P == best_m
    # m = best_m works if best_m <= max_m (which it is since P=27720 < 1000000)
    # But we might find an even better m by checking if there's a smaller valid m
    # Actually best_m is already in [0, P-1] so it's the smallest such m.
    
    # But wait - we should also verify: maybe a different m with the same residue
    # but larger value gives the same f. That doesn't matter - we want max f,
    # and all m with the same residue give the same f. So best_m (the residue itself)
    # is optimal.
    
    # However, let's also do a targeted refinement: check m values near the best
    # residue across different periods to make sure we haven't missed anything.
    # Actually since f depends only on m mod P, this is unnecessary.
    
    # Final safety: ensure best_m is a valid Python int in range
    best_m = int(best_m)
    if best_m < 0 or best_m > max_m:
        best_m = 0
    
    return best_m

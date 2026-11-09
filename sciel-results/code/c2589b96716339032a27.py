import time
import sys

# EVOLVE-BLOCK-START
P = 27720  # lcm(1,2,...,12)

def _precompute_masks():
    """For each residue r in [0, P), compute bitmask of which i (1..12) divide r."""
    masks = [0] * P
    for r in range(P):
        m = 0
        for i in range(1, 13):
            if r % i == 0:
                m |= (1 << (i - 1))
        masks[r] = m
    return masks

MASKS = _precompute_masks()

def _has_matching(m, L):
    """Check if interval (m, m+L] contains distinct b_1..b_12 with i | b_i."""
    if L < 12:
        return False
    base = m + 1  # first element of interval
    base_mod = base % P
    
    # For each i in 1..12, find positions j in [0, L) where (base + j) % i == 0
    # (base + j) % i == 0 iff j % i == (-base) % i
    # Since P is a multiple of i, base_mod % i == base % i
    # So j % i == (-base_mod) % i
    
    # Build list of candidate positions for each i
    candidates = []
    for i in range(1, 13):
        target = (-base_mod) % i
        # j in [0, L) with j % i == target
        # j = target, target + i, target + 2i, ... < L
        pos = []
        j = target
        while j < L:
            pos.append(j)
            j += i
        candidates.append(pos)
    
    # Check that each i has at least one candidate
    for i in range(12):
        if not candidates[i]:
            return False
    
    # DFS matching: assign each i to a unique position
    # Try to match i=0..11 to positions
    used = [False] * L
    result = [False]
    
    def dfs(idx):
        if result[0]:
            return
        if idx == 12:
            result[0] = True
            return
        for pos in candidates[idx]:
            if not used[pos]:
                used[pos] = True
                dfs(idx + 1)
                used[pos] = False
                if result[0]:
                    return
    
    dfs(0)
    return result[0]

def _compute_f(m):
    """Compute f(12, m): least positive L such that (m, m+L] has the matching."""
    # L must be at least 12
    for L in range(12, 200):  # f is at most 144 per problem statement, use 200 as safety
        if _has_matching(m, L):
            return L
    return 999  # should never reach here

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin (90s limit)
    
    best_m = 0
    best_f = 0
    
    # Since f is periodic with period P, we only need to check m in [0, P)
    for r in range(P):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        f_val = _compute_f(r)
        if f_val > best_f:
            best_f = f_val
            best_m = r
    
    return int(best_m)
# EVOLVE-BLOCK-END

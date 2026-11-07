import time
import math

# EVOLVE-BLOCK-START
# Strategy:
# 1. Exploit periodicity: f(12, m) depends only on m mod 27720.
# 2. Iterate through all 27720 residues, compute f(12, r) for each.
# 3. Find the residue r that maximizes f(12, r).
# 4. Map r to the largest m in [0, 1000000] with m % 27720 == r.
# 5. Use bitmask-based bipartite matching for speed.

PERIOD = 27720
N = 12
MAX_M = 1000000
TIME_LIMIT = 85  # seconds, leave 5s margin before 90s kill

def has_valid_set_fast(m, L):
    """
    Check if there is a perfect matching for indices 1..12 to distinct numbers
    in (m, m+L] such that i divides b_i.
    Uses bitmask-based Kuhn's algorithm for speed.
    """
    if L < N:
        return False
    
    # Precompute neighbors as bitmasks for each i in 1..12
    # neighbors[i-1] is a bitmask where bit j is set if (m+1+j) % i == 0
    neighbors = [0] * N
    start = m + 1
    for i in range(1, N + 1):
        rem = (-start) % i
        if rem < L:
            mask = 0
            for j in range(rem, L, i):
                mask |= (1 << j)
            neighbors[i - 1] = mask
        # If no valid positions for any i, immediate fail
        if neighbors[i - 1] == 0:
            return False
    
    # match_right[j] = u means right node j is matched to left node u
    match_right = [-1] * L
    
    def dfs(u, visited_mask):
        # u is 0..11, representing i = u+1
        avail = neighbors[u] & ~visited_mask
        while avail:
            # Get lowest set bit
            lsb = avail & (-avail)
            j = lsb.bit_length() - 1
            avail ^= lsb
            if match_right[j] == -1 or dfs(match_right[j], visited_mask | lsb):
                match_right[j] = u
                return True
        return False
    
    for u in range(N):
        if not dfs(u, 0):
            return False
    return True

def f_12(m):
    """
    Compute the least positive L such that (m, m+L] has a valid set.
    """
    lo, hi = 12, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if has_valid_set_fast(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    
    # 1. Find the best residue r in [0, PERIOD)
    best_r = 0
    best_val = 0
    
    # Iterate through all residues
    for r in range(PERIOD):
        # Safety check with margin
        if time.time() - start_time > TIME_LIMIT:
            break
        
        val = f_12(r)
        if val > best_val:
            best_val = val
            best_r = r
    
    # 2. Map best_r to the largest m in [0, MAX_M] with m % PERIOD == best_r
    if best_r > MAX_M:
        m = best_r
    else:
        k_max = (MAX_M - best_r) // PERIOD
        m = k_max * PERIOD + best_r
        
    # Ensure m is within bounds
    if m < 0:
        m = 0
    if m > MAX_M:
        m = MAX_M
        
    # Verify the result is a valid integer
    if not isinstance(m, int):
        m = int(m)
        
    return m
# EVOLVE-BLOCK-END

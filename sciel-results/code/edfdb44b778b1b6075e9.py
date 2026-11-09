"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _compute_lcm_range(n):
    result = 1
    for i in range(1, n + 1):
        result = _lcm(result, i)
    return result

def _has_perfect_matching(m, L, n=12):
    """Check if there's a perfect matching in interval (m, m+L] for indices 1..n."""
    # For each index i, find which positions j in 1..L have (m+j) % i == 0
    # Build adjacency: adj[i] = bitmask of valid positions
    adj = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, L + 1):
            if (m + j) % i == 0:
                mask |= (1 << (j - 1))
        adj[i] = mask
        if mask == 0:
            return False
    
    # Use DFS-based bipartite matching (Kuhn's algorithm)
    # match_pos[j] = which index i is matched to position j (1-indexed positions)
    match_pos = [-1] * (L + 1)  # match_pos[j] = i if position j is matched to index i
    
    def dfs(i, seen):
        for j in range(1, L + 1):
            if (adj[i] & (1 << (j - 1))) and not seen[j]:
                seen[j] = True
                if match_pos[j] == -1 or dfs(match_pos[j], seen):
                    match_pos[j] = i
                    return True
        return False
    
    for i in range(1, n + 1):
        seen = [False] * (L + 1)
        if not dfs(i, seen):
            return False
    return True

def _compute_f(m, n=12, max_L=144):
    """Compute f(n, m): the least positive L such that (m, m+L] contains distinct b_1,...,b_n with i dividing b_i."""
    for L in range(1, max_L + 1):
        if _has_perfect_matching(m, L, n):
            return L
    return max_L

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds, leaving 10s safety margin
    
    n = 12
    lcm_period = _compute_lcm_range(n)  # LCM(1..12) = 27720
    
    # Search over all residues m mod lcm_period
    # For each residue, compute f(n, m) and track the maximum
    best_m = 0
    best_L = 0
    
    # Precompute for each residue r in [0, lcm_period), the value of f(n, r)
    # Since the problem is periodic with period lcm_period, f(n, m) depends only on m mod lcm_period
    
    # We'll search all residues in the fundamental period
    for r in range(lcm_period):
        if time.time() > deadline:
            break
        L = _compute_f(r, n)
        if L > best_L:
            best_L = L
            best_m = r
    
    # Now we need to find an m in [0, 1000000] that has the same residue as best_m mod lcm_period
    # and maximizes f(n, m). Since f is periodic, any m ≡ best_m (mod lcm_period) will have the same f.
    # We just need to return a valid m in [0, 1000000].
    
    # The best residue is best_m. Find the smallest m >= 0 with m ≡ best_m (mod lcm_period)
    # That's just best_m itself if best_m <= 1000000, otherwise we need to check.
    
    if best_m <= 1000000:
        return int(best_m)
    else:
        # best_m is already reduced mod lcm_period, so best_m < lcm_period = 27720
        # So best_m is always <= 1000000
        return int(best_m)

# EVOLVE-BLOCK-END

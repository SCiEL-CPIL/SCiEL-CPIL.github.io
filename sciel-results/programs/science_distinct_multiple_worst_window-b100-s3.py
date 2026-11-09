"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import product

def _precompute_divisibility():
    """For each residue r in 0..27719 and offset j in 0..143, 
    compute bitmask of which i (1..12) divide (r+1+j)."""
    P = 27720
    MAX_L = 144
    # div_mask[r][j] = bitmask where bit i-1 is set if i divides (r+1+j)
    div_mask = [[0] * MAX_L for _ in range(P)]
    for r in range(P):
        for j in range(MAX_L):
            val = r + 1 + j
            mask = 0
            for i in range(1, 13):
                if val % i == 0:
                    mask |= (1 << (i - 1))
            div_mask[r][j] = mask
    return div_mask

def _has_matching(div_mask_r, L):
    """Check if there's a matching of size 12 in window of length L.
    div_mask_r[j] has bit i-1 set if i divides position j (0-indexed).
    We need to assign each i (1..12) to a distinct position j in 0..L-1 
    such that i divides position j's value."""
    # Use DFS with bitmasks
    # left: i from 1 to 12, right: positions 0 to L-1
    # adj[i] = set of positions where i divides
    adj = [0] * 13  # adj[i] is bitmask of positions
    for j in range(L):
        mask = div_mask_r[j]
        for i in range(1, 13):
            if mask & (1 << (i - 1)):
                adj[i] |= (1 << j)
    
    # DFS matching: try to match all 12 left nodes
    # Use recursive backtracking with memoization on (i, used_mask)
    # But L can be up to 144, so used_mask would need 144 bits - too large
    # Instead, use standard bipartite matching (Hopcroft-Karp or DFS augmenting paths)
    
    # Standard DFS-based bipartite matching
    match_to_pos = [-1] * L  # match_to_pos[j] = i matched to position j, or -1
    
    def dfs(i, visited):
        """Try to find augmenting path from left node i"""
        for j in range(L):
            if (adj[i] >> j) & 1 and not visited[j]:
                visited[j] = True
                if match_to_pos[j] == -1 or dfs(match_to_pos[j], visited):
                    match_to_pos[j] = i
                    return True
        return False
    
    count = 0
    for i in range(1, 13):
        visited = [False] * L
        if dfs(i, visited):
            count += 1
        else:
            return False
    return count == 12

def _compute_f(r, div_mask):
    """Compute f(12, m) where m mod 27720 = r."""
    for L in range(1, 145):
        if _has_matching(div_mask[r], L):
            return L
    return 145  # shouldn't happen

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    div_mask = _precompute_divisibility()
    
    best_m = 12  # fallback
    best_f = 12
    
    for r in range(27720):
        if time.time() - start_time > DEADLINE:
            break
        f_val = _compute_f(r, div_mask)
        if f_val > best_f:
            best_f = f_val
            # m = r works (r is in 0..27719, which is in 0..1000000)
            best_m = r
    
    return int(best_m)
# EVOLVE-BLOCK-END

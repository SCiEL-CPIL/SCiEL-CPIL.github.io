"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

# Precompute divisors of numbers 1..144 for each index i in 1..12
# For each i, the set of j in 1..144 such that i divides j
N = 12
MAX_L = 144

def _precompute_divisible():
    """For each index i (1..12), list which positions j (1..144) are divisible by i."""
    div = {}
    for i in range(1, N + 1):
        div[i] = [j for j in range(1, MAX_L + 1) if j % i == 0]
    return div

_DIV = _precompute_divisible()

def _has_perfect_matching(m, L):
    """
    Check if there exist distinct b_1,...,b_12 in (m, m+L] with i | b_i.
    We map this to: for each i, the set of positions j in 1..L such that (m+j) % i == 0.
    We need a perfect matching in the bipartite graph between indices {1..12} and positions {1..L}.
    """
    # Build adjacency: for each i, which positions j (1-indexed, 1..L) work
    adj = [None] * (N + 1)
    for i in range(1, N + 1):
        adj[i] = [j for j in range(1, L + 1) if (m + j) % i == 0]
    
    # Hungarian-style DFS matching
    match_to_pos = [-1] * (L + 1)  # match_to_pos[j] = i assigned to position j, or -1
    
    def dfs(i, visited):
        for j in adj[i]:
            if visited[j]:
                continue
            visited[j] = True
            if match_to_pos[j] == -1 or dfs(match_to_pos[j], visited):
                match_to_pos[j] = i
                return True
        return False
    
    for i in range(1, N + 1):
        visited = [False] * (L + 1)
        if not dfs(i, visited):
            return False
    return True

def _compute_f(m):
    """
    Compute f(12, m): the LEAST positive L such that (m, m+L] contains
    distinct b_1,...,b_12 with i | b_i.
    """
    for L in range(1, MAX_L + 1):
        if _has_perfect_matching(m, L):
            return L
    return MAX_L  # shouldn't happen for L up to 144

# LCM of 1..12 = 27720
LCM_12 = 27720

def construct():
    """
    Find m in 0..1000000 that maximizes f(12, m).
    Uses periodicity: f(12, m) depends only on m mod LCM_12.
    So we only need to check m in 0..LCM_12-1, then pick the best residue class.
    """
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    
    best_m = 12
    best_f = _compute_f(12)
    
    # We need to find the m in 0..1000000 that maximizes f(12,m).
    # Since f depends on m mod LCM_12, for each residue r in 0..LCM_12-1,
    # f(12, m) = f(12, r) for all m ≡ r (mod LCM_12).
    # The best m is the smallest m in 0..1000000 with the best residue.
    
    # Compute f for each residue
    best_residue = -1
    best_residue_f = 0
    
    for r in range(LCM_12):
        if time.time() > deadline:
            break
        f_val = _compute_f(r)
        if f_val > best_residue_f:
            best_residue_f = f_val
            best_residue = r
    
    # The best m is the smallest non-negative integer ≡ best_residue (mod LCM_12)
    # that is in 0..1000000. That's just best_residue itself (since 0 <= best_residue < LCM_12 <= 1000000).
    m = best_residue if best_residue >= 0 else 0
    
    # Verify m is in range
    if m < 0 or m > 1000000:
        m = 0
    
    return int(m)

# EVOLVE-BLOCK-END

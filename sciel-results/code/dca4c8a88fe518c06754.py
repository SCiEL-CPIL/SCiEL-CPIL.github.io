import time
from functools import lru_cache

# EVOLVE-BLOCK-START
P = {'n': 12, 'maximum_start': 1000000}
_LCM = 27720  # lcm(1..12)

def _has_perfect_matching(m, L, n=12):
    """Check if (m, m+L] contains n distinct b_i with i|b_i."""
    if L < n:
        return False
    # Build adjacency: for each i in 1..n, which numbers in (m, m+L] are divisible by i
    # Numbers are m+1, m+2, ..., m+L
    # For matching, we need to assign each i a distinct number
    # Use DFS-based bipartite matching
    # Left side: i=1..n, Right side: indices 0..L-1 (representing m+1+j)
    # adj[i] = list of right-side nodes j where (m+1+j) % i == 0
    adj = [[] for _ in range(n)]
    for j in range(L):
        num = m + 1 + j
        for i in range(1, n+1):
            if num % i == 0:
                adj[i-1].append(j)
    # Bipartite matching via DFS (Kuhn's algorithm)
    match_r = [-1] * L  # match_r[j] = left node matched to right node j, or -1
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_r[v] = u
                return True
        return False
    result = 0
    for u in range(n):
        visited = [False] * L
        if dfs(u, visited):
            result += 1
    return result == n

def _compute_f(m, n=12, max_L=144):
    """Compute f(n, m): least positive L such that (m, m+L] has n distinct b_i with i|b_i."""
    lo, hi = 1, max_L
    # First check if any L works
    if not _has_perfect_matching(m, max_L, n):
        # Try larger? But max_L=144 should be enough per task
        return max_L + 1  # sentinel
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    deadline = time.time() + 85  # safety margin before 90s
    n = P['n']
    max_start = P['maximum_start']
    
    best_f = 0
    best_m = 0
    
    # Search over residues mod LCM
    # For each residue r, compute f(12, r)
    # Then the best m in [0, max_start] with m % LCM == r is r itself (if r <= max_start)
    # or r + k*LCM for some k
    
    for r in range(_LCM):
        if time.time() > deadline:
            break
        f_val = _compute_f(r, n)
        if f_val > best_f:
            best_f = f_val
            # Find smallest m in [0, max_start] with m % LCM == r
            # m = r (since r < LCM and r <= max_start)
            best_m = r
    
    return int(best_m)
# EVOLVE-BLOCK-END

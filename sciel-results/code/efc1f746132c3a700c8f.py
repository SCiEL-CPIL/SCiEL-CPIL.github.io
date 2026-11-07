"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

P = {'n': 12, 'maximum_start': 1000000}

# LCM(1..12) = 27720, so f(12, m) depends only on m mod 27720
PERIOD = 27720

def has_valid_set(m, L):
    """Check if (m, m+L] contains distinct b_1,...,b_12 with i|b_i."""
    # Precompute for each index i (1..12), which positions in [0, L) are valid
    # Position j corresponds to number m+1+j
    # Valid if (m+1+j) % i == 0
    
    # For each left node i (0..11, representing index i+1),
    # find all right nodes j (0..L-1) where (m+1+j) % (i+1) == 0
    
    # Use Kuhn's algorithm (DFS-based bipartite matching)
    # Left: 0..11, Right: 0..L-1
    
    # Precompute adjacency lists
    adj = [[] for _ in range(12)]
    for i in range(12):
        idx = i + 1
        # Find j such that (m+1+j) % idx == 0
        # j ≡ -(m+1) mod idx, i.e., j ≡ (idx - (m+1) % idx) % idx
        r = (idx - (m + 1) % idx) % idx
        # j values: r, r+idx, r+2*idx, ... < L
        for j in range(r, L, idx):
            adj[i].append(j)
    
    match_to_right = [-1] * L
    
    def dfs(u, visited):
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                if match_to_right[v] == -1 or dfs(match_to_right[v], visited):
                    match_to_right[v] = u
                    return True
        return False
    
    for u in range(12):
        visited = [False] * L
        if not dfs(u, visited):
            return False
    return True

def f_12(m):
    """Find minimum L such that (m, m+L] has a valid set."""
    lo, hi = 12, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if has_valid_set(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # Compute f(12, r) for all r in 0..PERIOD-1
    # Since f(12, m) depends only on m mod PERIOD, we just need to check one representative
    # Use r itself as the representative (m = r)
    
    best_r = 0
    best_val = 0
    
    for r in range(PERIOD):
        if time.time() > deadline:
            # Fallback: return best found so far
            break
        val = f_12(r)
        if val > best_val:
            best_val = val
            best_r = r
    
    # Find the smallest m in [0, 1000000] such that m mod PERIOD == best_r
    # m = best_r, best_r + PERIOD, best_r + 2*PERIOD, ...
    # Smallest non-negative: m = best_r (if best_r <= 1000000)
    # Since PERIOD = 27720 and best_r < 27720, best_r is always <= 1000000
    
    m_result = best_r
    if m_result > 1000000:
        # This shouldn't happen since best_r < 27720
        m_result = 0
    
    return int(m_result)

# EVOLVE-BLOCK-END

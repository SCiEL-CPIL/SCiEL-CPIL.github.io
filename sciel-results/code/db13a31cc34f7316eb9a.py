"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _has_matching(m, L):
    """Check if (m, m+L] contains distinct b_1..b_12 with i|b_i using Kuhn's algorithm."""
    n = 12
    # Build adjacency: for each i (1..12), list of offsets j in [0, L-1] such that i | (m+1+j)
    adj = [[] for _ in range(n)]
    
    for i in range(1, n + 1):
        # We need k*i in (m, m+L]
        # k*i > m  =>  k > m/i  =>  k >= floor(m/i) + 1
        # k*i <= m+L => k <= (m+L)/i
        start_k = (m // i) + 1
        end_k = (m + L) // i
        for k in range(start_k, end_k + 1):
            val = k * i
            j = val - m - 1  # offset from m, 0-indexed (val = m + 1 + j)
            if 0 <= j < L:
                adj[i - 1].append(j)
        
        # Early termination: if any index has no candidates, matching is impossible
        if not adj[i - 1]:
            return False

    # Kuhn's algorithm (DFS augmenting paths)
    match_r = [-1] * L  # match_r[j] = i if right node j is matched to left node i
    
    # Pre-compute order of left nodes by degree (fewest candidates first) for better pruning
    # This is a heuristic to find augmenting paths faster, but standard Kuhn works fine too.
    # Since n=12 is small, standard Kuhn is very fast.
    
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_r[v] = u
                return True
        return False

    for u in range(n):
        visited = [False] * L
        if not dfs(u, visited):
            return False
    return True

def _compute_f(m):
    """Compute f(12, m): least L > 0 such that matching exists in (m, m+L]."""
    # Lower bound: for each i, the first multiple of i > m is at distance dist_i.
    # L must be at least max(dist_i) because b_i must be in (m, m+L] and divisible by i.
    min_L = 0
    for i in range(1, 13):
        next_mult = ((m // i) + 1) * i
        dist = next_mult - m
        if dist > min_L:
            min_L = dist
    
    # L must be at least 1
    if min_L < 1:
        min_L = 1
        
    for L in range(min_L, 145):
        if _has_matching(m, L):
            return L
    return 144 # Should not happen per contract, but safe fallback

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    n = 12
    max_m = 1000000
    period = 27720  # LCM(1..12)
    
    best_m = 0
    best_f = 0
    
    # f(12, m) is periodic with period 27720.
    # We only need to search m in [0, 27720).
    
    # Let's scan the entire period.
    for m in range(period):
        if time.time() > deadline:
            break
            
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            
    return int(best_m)
# EVOLVE-BLOCK-END

import time

P = 27720  # LCM(1..12)

def precompute_divisibility():
    """For each residue r in 0..P-1, store a bitmask of which i in 1..12 divide r."""
    div_mask = [0] * P
    for r in range(P):
        mask = 0
        for i in range(1, 13):
            if r % i == 0:
                mask |= (1 << (i - 1))
        div_mask[r] = mask
    return div_mask

DIV_MASK = precompute_divisibility()

def has_perfect_matching(m, L):
    """Check if (m, m+L] has a perfect matching for 1..12."""
    if L < 12:
        return False
    base = m + 1
    # Build adjacency: for each i (0..11), list of positions j where i+1 divides base+j
    adj = [[] for _ in range(12)]
    
    # Precompute residues for the window
    start_res = base % P
    for j in range(L):
        r = (start_res + j) % P
        mask = DIV_MASK[r]
        # For each i, if bit i is set, position j is available for i
        mm = mask
        while mm:
            i = (mm & (-mm)).bit_length() - 1
            mm &= mm - 1
            adj[i].append(j)
    
    # Quick check: each i must have at least one candidate
    for i in range(12):
        if not adj[i]:
            return False
    
    # Bipartite matching using DFS (Kuhn's algorithm)
    match_to = [-1] * L
    match_from = [-1] * 12
    
    def dfs(i, visited):
        for j in adj[i]:
            if visited[j]:
                continue
            visited[j] = True
            if match_to[j] == -1 or dfs(match_to[j], visited):
                match_to[j] = i
                match_from[i] = j
                return True
        return False
    
    for i in range(12):
        visited = [False] * L
        if not dfs(i, visited):
            return False
    
    return True

def compute_f(m):
    """Compute f(12, m) = least L such that (m, m+L] has perfect matching."""
    # Binary search for minimum L, lo=12 since we need 12 distinct integers
    lo, hi = 12, 144
    # Check hi works
    if not has_perfect_matching(m, hi):
        # Shouldn't happen per task constraints, but handle gracefully
        return hi
    while lo < hi:
        mid = (lo + hi) // 2
        if has_perfect_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    best_m = 0
    best_L = 0
    
    # Only need to check m in 0..P-1 since f(12, m) is periodic with period P
    # Check all residues 0..P-1
    
    # To speed up, process in chunks and check time
    chunk_size = 100
    for m_start in range(0, P, chunk_size):
        m_end = min(m_start + chunk_size, P)
        for m in range(m_start, m_end):
            if time.time() > deadline:
                return best_m
            
            L = compute_f(m)
            if L > best_L:
                best_L = L
                best_m = m
                if best_L >= 144:
                    # Can't do better
                    return best_m
        if time.time() > deadline:
            break
    
    return best_m

import time

PERIOD = 27720
N = 12
MAX_L = 144

def _can_match(adj, L):
    """
    Check if a perfect matching exists.
    adj: list of 12 lists, adj[u] contains valid offsets (1..L) for index u+1.
    L: current window size.
    """
    # Quick check
    for u in range(N):
        if not adj[u]:
            return False
            
    match_to_u = [-1] * (L + 1)
    
    def dfs(u, visited_mask):
        for off in adj[u]:
            if off > L:
                continue
            v_idx = off - 1
            bit = 1 << v_idx
            if visited_mask & bit:
                continue
            visited_mask_new = visited_mask | bit
            if match_to_u[off] == -1 or dfs(match_to_u[off], visited_mask_new):
                match_to_u[off] = u
                return True
        return False

    for u in range(N):
        if not dfs(u, 0):
            return False
    return True

def _f_from_cands(cands):
    """
    Find smallest L using precomputed candidates.
    cands: list of 12 lists of offsets (unsorted, but we filter by L in _can_match? 
           No, better to pre-filter or filter inside _can_match).
    Actually, _can_match takes L and uses adj. 
    We should pass the full list of possible offsets (up to MAX_L) and let _can_match 
    ignore those > L? Or filter beforehand?
    Filtering beforehand creates a new list, which has overhead.
    Checking `off <= L` inside the loop in _can_match is cheap.
    """
    # Binary search for L
    if not _can_match(cands, MAX_L):
        return MAX_L
        
    lo, hi = 1, MAX_L
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(cands, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    deadline = time.time() + 80
    
    best_L = 0
    best_r = 0
    
    for r in range(PERIOD):
        if time.time() > deadline:
            break
            
        # Precompute candidates for this r
        cands = []
        for i in range(1, N + 1):
            first = ((r // i) + 1) * i
            vals = []
            v = first
            while v <= r + MAX_L:
                vals.append(v - r)
                v += i
            cands.append(vals)
            
        f = _f_from_cands(cands)
        if f > best_L:
            best_L = f
            best_r = r
            
    return int(best_r)

import time

PERIOD = 27720
N = 12
MAX_L = 144

def _can_match(adj, L):
    """
    Check if a perfect matching exists for the given adjacency lists and window size L.
    adj: list of 12 lists. adj[u] contains offsets (integers) valid for index u+1.
    L: maximum offset allowed.
    """
    # Pre-filter or check on the fly? 
    # Checking on the fly in the DFS loop is efficient enough.
    
    match_to_u = [-1] * (L + 1)
    
    # Use a list for visited mask? No, integer bitmask is faster.
    # But we need to pass it recursively.
    
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
    """
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

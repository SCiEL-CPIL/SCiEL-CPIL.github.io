import time
from bisect import bisect_right

P = 27720  # LCM(1..12)
MAX_L = 144

def precompute_offsets():
    """Precompute for each residue r and each i, the sorted list of offsets j in 0..MAX_L-1 where i divides (r+j) mod P."""
    offsets = [[None] * 12 for _ in range(P)]
    for r in range(P):
        for i in range(12):
            lst = []
            for j in range(MAX_L):
                if (r + j) % (i + 1) == 0:
                    lst.append(j)
            offsets[r][i] = lst
    return offsets

OFFSETS = precompute_offsets()

def has_perfect_matching(m, L):
    """Check if (m, m+L] has a perfect matching for 1..12."""
    base_res = (m + 1) % P
    
    # Build adjacency: for each i, list of positions j < L where i+1 divides base+j
    adj = []
    for i in range(12):
        full_list = OFFSETS[base_res][i]
        idx = bisect_right(full_list, L - 1)
        adj.append(full_list[:idx])
    
    # Quick check: if any i has no available positions, fail immediately
    for i in range(12):
        if len(adj[i]) == 0:
            return False
    
    # Hall's condition quick check for small sets
    # Check all subsets of size 1-4 for Hall's violation
    # This is a necessary condition; if violated, no perfect matching exists
    
    # For each i, check if |N({i})| >= 1 (already done above)
    # For pairs, check |N({i,j})| >= 2
    # For triples, check |N({i,j,k})| >= 3
    # For quadruples, check |N({i,j,k,l})| >= 4
    
    # Build neighbor sets as bitmasks for fast union
    # Since L <= 144, we can use Python ints as bitmasks
    nsets = []
    for i in range(12):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        nsets.append(mask)
    
    # Check Hall's condition for subsets up to size 4
    # Size 1
    for i in range(12):
        if nsets[i] == 0:
            return False
    
    # Size 2
    for i in range(12):
        for j in range(i + 1, 12):
            union = nsets[i] | nsets[j]
            if bin(union).count('1') < 2:
                return False
    
    # Size 3
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                union = nsets[i] | nsets[j] | nsets[k]
                if bin(union).count('1') < 3:
                    return False
    
    # Size 4
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                for l in range(k + 1, 12):
                    union = nsets[i] | nsets[j] | nsets[k] | nsets[l]
                    if bin(union).count('1') < 4:
                        return False
    
    # Bipartite matching using DFS (Kuhn's algorithm)
    # Process left nodes in order of increasing degree (most constrained first)
    order = sorted(range(12), key=lambda i: len(adj[i]))
    
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
    
    for i in order:
        visited = [False] * L
        if not dfs(i, visited):
            return False
    
    return True

def fast_hall_check(m, L):
    """Fast necessary condition check using Hall's theorem for small subsets."""
    base_res = (m + 1) % P
    
    # Build neighbor sets as bitmasks
    nsets = []
    for i in range(12):
        full_list = OFFSETS[base_res][i]
        idx = bisect_right(full_list, L - 1)
        mask = 0
        for j in full_list[:idx]:
            mask |= (1 << j)
        nsets.append(mask)
        if mask == 0:
            return False
    
    # Check Hall's condition for subsets up to size 4
    # Size 2
    for i in range(12):
        for j in range(i + 1, 12):
            union = nsets[i] | nsets[j]
            if bin(union).count('1') < 2:
                return False
    
    # Size 3
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                union = nsets[i] | nsets[j] | nsets[k]
                if bin(union).count('1') < 3:
                    return False
    
    # Size 4
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                for l in range(k + 1, 12):
                    union = nsets[i] | nsets[j] | nsets[k] | nsets[l]
                    if bin(union).count('1') < 4:
                        return False
    
    return True

def compute_f(m):
    """Compute f(12, m) = least L such that (m, m+L] has perfect matching."""
    lo, hi = 12, 144
    if not has_perfect_matching(m, hi):
        hi = 200
        if not has_perfect_matching(m, hi):
            return hi
    while lo < hi:
        mid = (lo + hi) // 2
        if has_perfect_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def compute_f_fast(m):
    """Compute f(12, m) using fast Hall check first, then exact matching."""
    lo, hi = 12, 144
    if not fast_hall_check(m, hi):
        # If Hall fails at 144, try larger
        if not has_perfect_matching(m, 144):
            return 144
    while lo < hi:
        mid = (lo + hi) // 2
        if fast_hall_check(m, mid) and has_perfect_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_m = 0
    best_L = 0
    
    # Phase 1: Fast screening with Hall's condition
    # For each residue, quickly check if it could have high f(12,m)
    # A residue has high f(12,m) if Hall's condition fails for large L
    
    # First pass: use fast_hall_check to identify candidates
    # For each L from 144 down to 12, find residues where Hall fails at L
    # but succeeds at L-1 (or use a simpler heuristic)
    
    # Simpler approach: just do the full search but with the fast pre-check
    for m_start in range(0, P, 200):
        m_end = min(m_start + 200, P)
        for m in range(m_start, m_end):
            if time.time() > deadline:
                break
            
            L = compute_f_fast(m)
            if L > best_L:
                best_L = L
                best_m = m
                if best_L >= 144:
                    return int(best_m)
        if time.time() > deadline:
            break
    
    return int(best_m)

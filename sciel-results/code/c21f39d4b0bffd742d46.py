import time

def construct():
    P = 27720  # LCM(1..12)
    max_L = 144
    best_m = 0
    best_f = 0
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    def has_matching(m, L):
        # Build adjacency masks
        adj = [0] * 12
        adj_deg = [0] * 12
        for i in range(1, 13):
            d = i
            start = (-(m + 1)) % d
            mask = 0
            count = 0
            j = start
            while j < L:
                mask |= (1 << j)
                count += 1
                j += d
            adj[i - 1] = mask
            adj_deg[i - 1] = count
        
        # Order left nodes by degree ascending (fewest options first)
        order = sorted(range(12), key=lambda v: adj_deg[v])
        
        # Kuhn's algorithm
        match = [-1] * L  # match[j] = i
        
        def try_kuhn(v, visited):
            mask = adj[v]
            m_mask = mask
            while m_mask:
                lsb = m_mask & (-m_mask)
                j = lsb.bit_length() - 1
                m_mask ^= lsb
                
                if visited & (1 << j):
                    continue
                visited |= (1 << j)
                
                if match[j] == -1 or try_kuhn(match[j], visited):
                    match[j] = v
                    return True
            return False
        
        for i in order:
            if not try_kuhn(i, 0):
                return False
        
        return True

    def compute_f(m):
        # Binary search for minimum L where matching exists
        lo, hi = 12, max_L
        while lo < hi:
            mid = (lo + hi) // 2
            if has_matching(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    for m in range(P):
        if time.time() > deadline:
            break
        f_val = compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m

    return best_m

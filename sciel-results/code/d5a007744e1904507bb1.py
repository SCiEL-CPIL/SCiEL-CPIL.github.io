import time

def construct():
    P = 27720  # LCM(1..12)
    max_L = 144
    best_m = 0
    best_f = 0
    
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    def has_matching(m, L):
        # Build adjacency masks
        adj = [0] * 12
        for i in range(1, 13):
            d = i
            # We need (m + 1 + j) % d == 0
            # j % d == (-(m + 1)) % d
            start = (-(m + 1)) % d
            mask = 0
            j = start
            while j < L:
                mask |= (1 << j)
                j += d
            adj[i - 1] = mask
        
        # Kuhn's algorithm
        match = [-1] * L  # match[j] = i
        
        def try_kuhn(v, visited):
            # v is left node index (0..11)
            # visited is bitmask of right nodes visited in this DFS
            mask = adj[v]
            # Iterate through set bits in mask
            m_mask = mask
            while m_mask:
                # Get lowest set bit
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
        
        matched_count = 0
        for i in range(12):
            if try_kuhn(i, 0):
                matched_count += 1
        
        return matched_count == 12

    def compute_f(m):
        L = 12
        while L <= max_L:
            if has_matching(m, L):
                return L
            L += 1
        return max_L  # Should not reach if max is 144, but safe

    for m in range(P):
        if time.time() > deadline:
            break
        f_val = compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m

    return best_m

def has_independent_set_10(adj, n, all_mask, non_adj):
    # Greedy check
    mask = all_mask
    size = 0
    while mask:
        if size >= 10:
            return True
        
        min_deg = n + 1
        best_v = -1
        temp = mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (adj[v] & mask).bit_count()
            if deg < min_deg:
                min_deg = deg
                best_v = v
            temp ^= lsb
            
        if best_v == -1:
            break
            
        size += 1
        mask &= non_adj[best_v]
        
    if size >= 10:
        return True
        
    # Exact check
    def dfs(mask, size):
        if size == 10:
            return True
        if mask == 0:
            return False
        if mask.bit_count() + size < 10:
            return False
            
        max_deg = -1
        best_v = -1
        temp = mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (adj[v] & mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= lsb
            
        if best_v == -1:
            return False
            
        v_bit = 1 << best_v
        
        if dfs(non_adj[best_v] & mask, size + 1):
            return True
        
        if dfs(mask & ~v_bit, size):
            return True
            
        return False
        
    return dfs(all_mask, 0)

import time

def construct():
    start_time = time.time()
    limit = 85  # Leave 5s for overhead
    
    N = 12
    period = 27720
    max_L = 150
    
    # Precompute masks for each i, residue r, and L
    # precomputed[i][r][L] is a bitmask of positions j in [0, L) such that j % i == r
    precomputed = {}
    for i in range(1, N + 1):
        precomputed[i] = {}
        for r in range(i):
            masks_for_L = [0] * (max_L + 1)
            cur_mask = 0
            # We can fill this incrementally
            for L in range(1, max_L + 1):
                j = L - 1
                if j % i == r:
                    cur_mask |= (1 << j)
                masks_for_L[L] = cur_mask
            precomputed[i][r] = masks_for_L
    
    def compute_f(m):
        residues = [0] * (N + 1)
        for i in range(1, N + 1):
            residues[i] = (-m - 1) % i
        
        for L in range(1, max_L + 1):
            masks = [0] * (N + 1)
            union_mask = 0
            valid = True
            for i in range(1, N + 1):
                mask = precomputed[i][residues[i]][L]
                if mask == 0:
                    valid = False
                    break
                masks[i] = mask
                union_mask |= mask
            
            if not valid:
                continue
                
            if union_mask.bit_count() < N:
                continue
            
            # Order indices by popcount of their mask
            indices = list(range(1, N + 1))
            indices.sort(key=lambda i: masks[i].bit_count())
            
            # Backtracking
            def backtrack(idx, used):
                if idx == N:
                    return True
                i = indices[idx]
                avail = masks[i] & ~used
                while avail:
                    bit = avail & -avail
                    avail ^= bit
                    if backtrack(idx + 1, used | bit):
                        return True
                return False
            
            if backtrack(0, 0):
                return L
        
        return max_L
    
    best_m = 0
    best_f = 0
    
    for m in range(period):
        if time.time() - start_time > limit:
            break
        
        f_val = compute_f(m)
        
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    return int(best_m)

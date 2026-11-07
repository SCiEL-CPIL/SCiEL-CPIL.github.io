import time

PERIOD = 27720
N = 12
MAX_M = 1000000
MAX_L = 144

def construct():
    start_time = time.time()
    
    best_r = 0
    best_val = 0
    
    for r in range(PERIOD):
        # Precompute base neighbors for L=MAX_L
        base_neighbors = [0] * N
        counts = [0] * N
        for i in range(1, N + 1):
            rem = (-r) % i
            first_k = rem if rem > 0 else i
            if first_k > MAX_L:
                # Should not happen for MAX_L=144
                base_neighbors[i-1] = 0
                counts[i-1] = 0
                continue
            
            mask = 0
            count = 0
            for k in range(first_k, MAX_L + 1, i):
                mask |= (1 << (k - 1))
                count += 1
            base_neighbors[i-1] = mask
            counts[i-1] = count
            
        # If any count is 0, it means no matching possible even at MAX_L?
        # But we know L=144 is always sufficient.
        
        # Binary search for minimal L
        lo, hi = 12, MAX_L
        while lo < hi:
            mid = (lo + hi) // 2
            # Check if matching exists for L=mid
            # Mask base_neighbors with (1 << mid) - 1
            
            # Quick check: if any (base_neighbors[i] & mask) == 0, fail
            mask_l = (1 << mid) - 1
            valid = True
            for i in range(N):
                if (base_neighbors[i] & mask_l) == 0:
                    valid = False
                    break
            
            if valid:
                # Run matching
                neighbors = [base_neighbors[i] & mask_l for i in range(N)]
                
                # Order by count of bits in neighbors
                # We can approximate count or just use fixed order?
                # Fixed order might be faster than sorting if sorting overhead is high.
                # But sorting 12 items is very fast.
                # Let's use the precomputed counts? No, counts change with L.
                # We can compute bit counts quickly.
                
                # Actually, let's just use the Kuhn's algorithm with the sorted order from before?
                # The order depends on L.
                # Let's just do a simple DFS without sorting first to see if it's fast enough.
                # Or sort by number of set bits.
                
                # Helper for popcount
                def popcount(x):
                    return bin(x).count('1')
                
                order = sorted(range(N), key=lambda idx: popcount(neighbors[idx]))
                
                match_right = [-1] * mid
                
                def dfs(u, visited_mask):
                    avail = neighbors[u] & ~visited_mask
                    while avail:
                        lsb = avail & (-avail)
                        j = lsb.bit_length() - 1
                        avail ^= lsb
                        if match_right[j] == -1 or dfs(match_right[j], visited_mask | lsb):
                            match_right[j] = u
                            return True
                    return False
                
                match_found = True
                for u in order:
                    if not dfs(u, 0):
                        match_found = False
                        break
                        
                if match_found:
                    hi = mid
                else:
                    lo = mid + 1
            else:
                lo = mid + 1
                
        val = lo
        if val > best_val:
            best_val = val
            best_r = r
            
        if time.time() - start_time > 85:
            break

    # Map best_r to m
    if best_r > MAX_M:
        m = best_r
    else:
        k_max = (MAX_M - best_r) // PERIOD
        m = k_max * PERIOD + best_r
        
    if m < 0:
        m = 0
    if m > MAX_M:
        m = MAX_M
        
    return int(m)

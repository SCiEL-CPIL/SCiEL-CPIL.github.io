import time
import random

def count_c4s_bitmask(n, adj):
    """
    Count C4s using bitmask adjacency lists.
    adj is a list of 5 lists, each of length n, containing integer bitmasks.
    """
    total_c4s = 0
    for c in range(5):
        for u in range(n):
            mask_u = adj[c][u]
            for v in range(u + 1, n):
                mask_v = adj[c][v]
                common = mask_u & mask_v
                # Count bits in common
                # Python 3.8+ has int.bit_count()
                k = common.bit_count()
                if k >= 2:
                    total_c4s += k * (k - 1) // 2
    return total_c4s

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_n = 11
    best_colors = [0] * (11 * 10 // 2)  # Fallback
    
    for n in range(26, 10, -1):
        if time.time() > deadline:
            break
            
        num_edges = n * (n - 1) // 2
        # Initialize random
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Build initial bitmasks
        adj = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
                
        current_c4s = count_c4s_bitmask(n, adj)
        
        if current_c4s == 0:
            best_n = n
            best_colors = colors
            break
            
        max_iters = 200000
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Pick random edge
            idx = random.randint(0, num_edges - 1)
            
            # Determine vertices from index
            # Formula: idx = i*(n-1) - i*(i-1)//2 + (j-i-1)
            # Inverse is tricky. Precompute edge endpoints.
            # For speed, precompute endpoints for all edges once per n.
            # But n changes. We are inside the n loop.
            # Let's precompute endpoints for this n.
            # This is O(n^2) memory, which is fine for n=26.
            
            # Optimization: Precompute endpoints outside the inner loop?
            # Yes, move this up.
            
            if not hasattr(construct, '_endpoints_cache'):
                construct._endpoints_cache = {}
            if n not in construct._endpoints_cache:
                endpoints = []
                idx = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        endpoints.append((i, j))
                        idx += 1
                construct._endpoints_cache[n] = endpoints
            endpoints = construct._endpoints_cache[n]
            
            u, v = endpoints[idx]
            old_c = colors[idx]
            new_c = random.randint(0, 4)
            if old_c == new_c:
                continue
                
            # Update bitmasks
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            adj[new_c][u] |= (1 << v)
            adj[new_c][v] |= (1 << u)
            
            # Recount C4s
            new_c4s = count_c4s_bitmask(n, adj)
            
            if new_c4s < current_c4s:
                current_c4s = new_c4s
                colors[idx] = new_c
                if current_c4s == 0:
                    best_n = n
                    best_colors = colors
                    break
            else:
                # Revert
                adj[old_c][u] |= (1 << v)
                adj[old_c][v] |= (1 << u)
                adj[new_c][u] &= ~(1 << v)
                adj[new_c][v] &= ~(1 << u)
                # colors[idx] remains old_c
            
        if current_c4s == 0:
            break

    return {'n': best_n, 'colors': best_colors}

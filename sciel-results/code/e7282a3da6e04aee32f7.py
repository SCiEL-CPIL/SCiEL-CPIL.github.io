import time
import random
import numpy as np

def construct():
    """
    Returns {'n': n, 'colors': [...]} where colors is a list of length n*(n-1)//2
    giving one colour in {0,...,4} for each pair (i,j), i<j, in lexicographic order.
    No colour class may contain a C4 (no two vertices with >=2 common neighbours in one colour).
    """
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin before 150s kill

    # Helper to get edge index
    def edge_idx(i, j, n):
        return i * n - i * (i + 1) // 2 - 1 + (j - i - 1)

    def idx_to_edge(idx, n):
        # Binary search or linear scan. Linear is fine for n<=30.
        row_start = 0
        for i in range(n):
            row_len = n - 1 - i
            if idx < row_start + row_len:
                j = i + 1 + (idx - row_start)
                return (i, j)
            row_start += row_len
        return (0, 1)

    def count_c4_fast(n, colors):
        """
        Count C4 violations using bitmasks.
        Returns the total number of C4s (or a proxy metric).
        A C4 exists if there are two vertices u, v with >= 2 common neighbors in a color.
        We sum max(0, common_count - 1) for all pairs u,v and colors.
        """
        # adj[c][u] is a bitmask of neighbors of u in color c
        adj = [[0] * n for _ in range(5)]
        
        for idx in range(len(colors)):
            i, j = idx_to_edge(idx, n)
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            
        total = 0
        for c in range(5):
            # For each pair of vertices u, v, count common neighbors
            # common = adj[c][u] & adj[c][v]
            # If popcount(common) >= 2, it's a violation.
            # Actually, we just need to know if it's 0 or not.
            # To be efficient, we can just check if (adj[c][u] & adj[c][v]) has at least 2 bits set.
            # Or simpler: if (adj[c][u] & adj[c][v]) & ((adj[c][u] & adj[c][v]) - 1) != 0
            # This checks if there are at least 2 common neighbors.
            for u in range(n):
                mask_u = adj[c][u]
                if mask_u == 0: continue
                for v in range(u + 1, n):
                    mask_v = adj[c][v]
                    if mask_v == 0: continue
                    common = mask_u & mask_v
                    # Check if common has at least 2 bits
                    if common & (common - 1):
                        total += 1
        return total

    def local_search(n, time_limit):
        """
        Local search to minimize C4 count.
        """
        m = n * (n - 1) // 2
        # Initialization: structured
        colors = [0] * m
        for idx in range(m):
            i, j = idx_to_edge(idx, n)
            d = (j - i) % n
            colors[idx] = d % 5
            
        current_cost = count_c4_fast(n, colors)
        if current_cost == 0:
            return colors, True
            
        # Precompute edge lists for quick access
        edges = [idx_to_edge(idx, n) for idx in range(m)]
        
        # Build initial adjacency bitmasks
        adj = [[0] * n for _ in range(5)]
        for idx in range(m):
            i, j = edges[idx]
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            
        t_start = time.time()
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() - t_start < time_limit:
            if current_cost == 0:
                return colors, True
                
            # Pick a random edge to change
            eidx = random.randint(0, m - 1)
            i, j = edges[eidx]
            old_c = colors[eidx]
            
            # Try changing to a different color
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
                
            # Calculate delta cost
            # Removing edge (i,j) from color old_c
            # Adding edge (i,j) to color new_c
            
            # We need to update the cost.
            # Cost contribution of a pair (u,v) in color c is 1 if common_neighbors(u,v) >= 2.
            # This is hard to do incrementally without a global recount if we don't track counts of common neighbors.
            # Let's just do a fast recount for now, but optimized.
            # Or, we can track common neighbor counts.
            
            # For now, let's stick to a slightly slower but simpler incremental check
            # or just use the fast bitmask recount which is quite fast for n=26.
            
            # Apply change
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            adj[new_c][i] |= (1 << j)
            adj[new_c][j] |= (1 << i)
            colors[eidx] = new_c
            
            # Recount
            new_cost = count_c4_fast(n, colors)
            
            if new_cost < current_cost:
                current_cost = new_cost
                no_improve_count = 0
            else:
                # Revert
                adj[old_c][i] |= (1 << j)
                adj[old_c][j] |= (1 << i)
                adj[new_c][i] &= ~(1 << j)
                adj[new_c][j] &= ~(1 << i)
                colors[eidx] = old_c
                no_improve_count += 1
                
            if no_improve_count > max_no_improve:
                # Restart with random perturbation
                for _ in range(10):
                    eidx2 = random.randint(0, m - 1)
                    i2, j2 = edges[eidx2]
                    old_c2 = colors[eidx2]
                    new_c2 = random.randint(0, 4)
                    if new_c2 != old_c2:
                        adj[old_c2][i2] &= ~(1 << j2)
                        adj[old_c2][j2] &= ~(1 << i2)
                        adj[new_c2][i2] |= (1 << j2)
                        adj[new_c2][j2] |= (1 << i2)
                        colors[eidx2] = new_c2
                current_cost = count_c4_fast(n, colors)
                no_improve_count = 0

        return colors, (current_cost == 0)

    # Try n=26
    if time.time() < deadline:
        colors26, valid26 = local_search(26, time_limit=60.0)
        if valid26:
            return {'n': 26, 'colors': colors26}
            
    # Try n=25
    if time.time() < deadline:
        colors25, valid25 = local_search(25, time_limit=30.0)
        if valid25:
            return {'n': 25, 'colors': colors25}
            
    # Fallback n=20
    if time.time() < deadline:
        colors20, valid20 = local_search(20, time_limit=10.0)
        if valid20:
            return {'n': 20, 'colors': colors20}

    # Final fallback n=15
    colors15, _ = local_search(15, time_limit=5.0)
    return {'n': 15, 'colors': colors15}

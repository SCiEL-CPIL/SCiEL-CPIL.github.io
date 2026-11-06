import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try to find a solution for n=17
    for n in [17, 16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() > deadline - 2.0:
            break
        res = _local_search(n, deadline)
        if res is not None:
            return res
            
    # Fallback
    return _fallback()

def _fallback():
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append((i + j) % 4)
    return {'n': n, 'colors': colors}

def _local_search(n, deadline):
    """
    Local search to minimize the number of monochromatic C6s.
    """
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Precompute edge index
    edge_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx
        
    best_colors = None
    best_count = float('inf')
    
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        rng = random.Random(attempt * 12345)
        colors = [rng.randint(0, 3) for _ in range(num_edges)]
        
        # Build adjacency lists for each color
        adj = [[0] * n for _ in range(4)]
        for idx, (u, v) in enumerate(edges):
            c = colors[idx]
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            
        count = _count_c6s(n, adj)
        
        if count == 0:
            return {'n': n, 'colors': colors}
            
        if count < best_count:
            best_count = count
            best_colors = colors[:]
            
        # Local search steps
        steps = 0
        max_steps = 10000
        while steps < max_steps and time.time() < deadline:
            steps += 1
            if count == 0:
                return {'n': n, 'colors': colors}
                
            # Pick a random edge involved in a C6
            # For simplicity, pick a random edge and try to improve
            idx = rng.randint(0, num_edges - 1)
            u, v = edges[idx]
            old_c = colors[idx]
            
            best_c = old_c
            min_count = count
            
            for new_c in range(4):
                if new_c == old_c:
                    continue
                
                # Remove old edge
                adj[old_c][u] &= ~(1 << v)
                adj[old_c][v] &= ~(1 << u)
                
                # Add new edge
                adj[new_c][u] |= (1 << v)
                adj[new_c][v] |= (1 << u)
                
                new_count = _count_c6s(n, adj)
                
                # Revert
                adj[new_c][u] &= ~(1 << v)
                adj[new_c][v] &= ~(1 << u)
                
                adj[old_c][u] |= (1 << v)
                adj[old_c][v] |= (1 << u)
                
                if new_count < min_count:
                    min_count = new_count
                    best_c = new_c
                    
            if best_c != old_c:
                # Commit the change
                colors[idx] = best_c
                adj[old_c][u] &= ~(1 << v)
                adj[old_c][v] &= ~(1 << u)
                adj[best_c][u] |= (1 << v)
                adj[best_c][v] |= (1 << u)
                count = min_count
            else:
                # Stuck, do a random flip
                idx = rng.randint(0, num_edges - 1)
                u, v = edges[idx]
                old_c = colors[idx]
                new_c = rng.randint(0, 3)
                if new_c != old_c:
                    colors[idx] = new_c
                    adj[old_c][u] &= ~(1 << v)
                    adj[old_c][v] &= ~(1 << u)
                    adj[new_c][u] |= (1 << v)
                    adj[new_c][v] |= (1 << u)
                    count = _count_c6s(n, adj)
                    
    if best_colors is not None and best_count == 0:
        return {'n': n, 'colors': best_colors}
        
    return None

def _count_c6s(n, adj):
    """
    Count the number of monochromatic C6s.
    """
    total = 0
    for c in range(4):
        # Count C6s in color c
        # A C6 is a sequence u-v-w-x-y-z-u
        # We can count by iterating over pairs (u,v) and checking for paths of length 4
        # Or use the formula: number of closed walks of length 6 / 12
        # But we need simple cycles.
        # For n=17, we can just check all 6-tuples? No, too slow.
        # We can check for each pair (u,v) if there is a path of length 4.
        # This is O(n^3) per color if done naively.
        # Let's do a simple check: for each u, v, check if there is a path u-x1-x2-x3-x4-v
        # We can use bitsets.
        for u in range(n):
            for v in range(u + 1, n):
                # Check if there is a path of length 4 from u to v in adj[c]
                # u - x1 - x2 - x3 - x4 - v
                # x1 in N(u), x4 in N(v)
                nu = adj[c][u]
                nv = adj[c][v]
                if nu == 0 or nv == 0:
                    continue
                
                # We need to check if there is a path of length 2 from x1 to x4
                # for some x1 in N(u) and x4 in N(v)
                # Let's iterate x1
                m1 = nu
                while m1:
                    b1 = m1 & (-m1)
                    x1 = b1.bit_length() - 1
                    m1 ^= b1
                    
                    # Iterate x4
                    m4 = nv
                    while m4:
                        b4 = m4 & (-m4)
                        x4 = b4.bit_length() - 1
                        m4 ^= b4
                        
                        if x4 == u or x4 == x1:
                            continue
                        
                        # Check if there is a path of length 2 from x1 to x4
                        # x1 - x2 - x4
                        # x2 in N(x1) and x2 in N(x4)
                        # N(x1) & N(x4)
                        common = adj[c][x1] & adj[c][x4]
                        # Exclude u and v
                        common &= ~(1 << u)
                        common &= ~(1 << v)
                        # Exclude x1 and x4
                        common &= ~(1 << x1)
                        common &= ~(1 << x4)
                        
                        if common:
                            # Found a C6
                            total += 1
                            
    return total

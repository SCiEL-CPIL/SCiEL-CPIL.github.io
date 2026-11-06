import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Construct Polarity Graph of PG(2,4)
    # Field F4
    add = [[0,1,2,3], [1,0,3,2], [2,3,0,1], [3,2,1,0]]
    mul = [[0,0,0,0], [0,1,2,3], [0,2,3,1], [0,3,1,2]]
    
    def f4_add(a, b): return add[a][b]
    def f4_mul(a, b): return mul[a][b]
    def f4_sub(a, b): return add[a][b] # Char 2
    
    # Generate points of PG(2,4)
    points = []
    for x in range(4):
        for y in range(4):
            for z in range(4):
                if x==0 and y==0 and z==0: continue
                # Normalize
                if x != 0:
                    # x is 1, 2, or 3. Make it 1.
                    # inv(x)
                    if x == 1: inv = 1
                    elif x == 2: inv = 3
                    else: inv = 2
                    p = [1, f4_mul(y, inv), f4_mul(z, inv)]
                elif y != 0:
                    if y == 1: inv = 1
                    elif y == 2: inv = 3
                    else: inv = 2
                    p = [0, 1, f4_mul(z, inv)]
                else:
                    # z != 0
                    if z == 1: inv = 1
                    elif z == 2: inv = 3
                    else: inv = 2
                    p = [0, 0, 1]
                
                # Check if already in list (should be unique with normalization)
                if p not in points:
                    points.append(p)
    
    # points should have 21 elements
    n_pts = len(points)
    
    # Bilinear form B(x,y) = x0*y0 + x1*y2 + x2*y1
    def B(p, q):
        return f4_add(f4_add(f4_mul(p[0], q[0]), f4_mul(p[1], q[2])), f4_mul(p[2], q[1]))
    
    # Build adjacency for 21 vertices
    adj_21 = [0] * n_pts
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            if B(points[i], points[j]) == 0:
                adj_21[i] |= (1 << j)
                adj_21[j] |= (1 << i)
    
    # Delete vertex 0
    # Remap indices 1..20 to 0..19
    adj = [0] * 20
    for i in range(1, n_pts):
        new_i = i - 1
        # Mask out bit 0
        mask = adj_21[i] & ~(1 << 0)
        # Shift down bits > 0
        # Bit j in adj_21[i] corresponds to vertex j-1 in new graph
        # So we need to shift right by 1?
        # If bit j is set, new bit is j-1.
        # So mask >> 1
        adj[new_i] = mask >> 1
        
    # Local Search
    def is_valid_add(u, v, adj):
        # Check if edge exists
        if (adj[u] >> v) & 1:
            return False
        
        # Check common neighbors of u and v
        common_uv = adj[u] & adj[v]
        if bin(common_uv).count("1") > 2:
            return False # Should not happen in valid graph, but check anyway
            
        # Check neighbors of v (excluding u)
        # For each w in adj[v], w != u, check common(u, w)
        # common(u, w) = adj[u] & adj[w]
        # If count >= 2, adding (u,v) makes it 3?
        # Wait, if common(u,w) is 2, adding (u,v) adds v to adj[u].
        # Is v in adj[w]? Yes, because w in adj[v] implies v in adj[w].
        # So v becomes a new common neighbor. Count becomes 3. Invalid.
        # So we need common(u, w) <= 1 for all w in adj[v] \ {u}
        
        # Get neighbors of v
        neighbors_v = adj[v]
        # Exclude u
        neighbors_v &= ~(1 << u)
        
        # Iterate bits in neighbors_v
        temp = neighbors_v
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            common_uw = adj[u] & adj[w]
            if bin(common_uw).count("1") > 1:
                return False
            temp ^= lsb
            
        # Check neighbors of u (excluding v)
        neighbors_u = adj[u]
        neighbors_u &= ~(1 << v)
        
        temp = neighbors_u
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            common_vw = adj[v] & adj[w]
            if bin(common_vw).count("1") > 1:
                return False
            temp ^= lsb
            
        return True

    def count_edges(adj):
        count = 0
        for i in range(20):
            count += bin(adj[i]).count("1")
        return count // 2

    best_edges = count_edges(adj)
    best_adj = adj[:]
    
    # Greedy addition in random order, multiple times
    for _ in range(100):
        if time.time() > deadline:
            break
            
        current_adj = best_adj[:]
        current_edges = best_edges
        
        # Get list of missing edges
        missing = []
        for u in range(20):
            for v in range(u+1, 20):
                if not ((current_adj[u] >> v) & 1):
                    missing.append((u, v))
        
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline:
                break
            if is_valid_add(u, v, current_adj):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges += 1
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = current_adj[:]
            
    # Convert to list of edges
    edges = []
    for u in range(20):
        mask = best_adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            mask ^= lsb
            
    return edges

# EVOLVE-BLOCK-END

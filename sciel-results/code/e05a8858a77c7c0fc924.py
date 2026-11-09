import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def is_q3_mask(mask, adj):
        # Check if induced subgraph on mask is Q3
        # 1. Count vertices
        # We know mask has 8 bits set from caller
        
        # 2. Check degrees
        # Iterate bits
        m = mask
        edge_count = 0
        while m:
            # Get lowest set bit
            lsb = m & -m
            i = lsb.bit_length() - 1
            m ^= lsb
            
            # Count neighbors in mask
            deg = bin(adj[i] & mask).count("1")
            if deg != 3:
                return False
            edge_count += deg
            
        if edge_count != 24:
            return False
            
        # 3. Check bipartiteness
        # Get list of vertices
        verts = []
        m = mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            m ^= lsb
            verts.append(i)
            
        # BFS
        color = [-1] * 16
        color[verts[0]] = 0
        queue = [verts[0]]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            # Neighbors of u in mask
            neighbors_mask = adj[u] & mask
            nm = neighbors_mask
            while nm:
                lsb = nm & -nm
                v = lsb.bit_length() - 1
                nm ^= lsb
                
                if color[v] == color[u]:
                    return False
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    queue.append(v)
                    
        return True

    def has_q3_with_edge(u, v, adj):
        # Temporarily add edge
        adj_temp = adj[:]
        adj_temp[u] |= (1 << v)
        adj_temp[v] |= (1 << u)
        
        # Generate 6-subsets of others
        others = [i for i in range(16) if i != u and i != v]
        
        for comb in itertools.combinations(others, 6):
            mask = (1 << u) | (1 << v)
            for w in comb:
                mask |= (1 << w)
            if is_q3_mask(mask, adj_temp):
                return True
        return False

    best_edges = []
    best_count = 0
    
    # Initial graph: K3,13
    adj = [0] * 16
    for i in range(3):
        for j in range(3, 16):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    
    edges = [[i, j] for i in range(3) for j in range(3, 16)]
    best_edges = edges[:]
    best_count = len(edges)
    
    all_edges = [(i, j) for i in range(16) for j in range(i+1, 16)]
    
    # Greedy addition with random restarts
    while time.time() < deadline:
        # Reset to base graph
        adj = [0] * 16
        for i in range(3):
            for j in range(3, 16):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        current_edges = set([(i, j) for i in range(3) for j in range(3, 16)])
        
        missing = [e for e in all_edges if e not in current_edges]
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline:
                break
            if has_q3_with_edge(u, v, adj):
                continue
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.add((u, v))
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = [list(e) for e in current_edges]
            
        if best_count >= 52:
            break

    return best_edges

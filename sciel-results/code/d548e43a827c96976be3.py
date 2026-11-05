import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    best_edges = []
    best_count = 0
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_set = set(all_edges)
    
    def is_3_colorable(adj):
        # adj is list of bitmasks
        degs = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda i: degs[i], reverse=True)
        
        colors = [-1] * n
        color_masks = [0, 0, 0]
        
        def backtrack(idx_in_order):
            if idx_in_order == n:
                return True
            
            v = order[idx_in_order]
            
            avail = []
            for c in range(3):
                if (adj[v] & color_masks[c]) == 0:
                    avail.append(c)
                    
            if not avail:
                return False
            
            for c in avail:
                colors[v] = c
                color_masks[c] |= (1 << v)
                
                if backtrack(idx_in_order + 1):
                    return True
                
                colors[v] = -1
                color_masks[c] &= ~(1 << v)
            
            return False
        
        return backtrack(0)

    def try_add_edges(adj, edge_count):
        """Try to add more edges while maintaining triangle-freeness and non-3-colorability."""
        if time.time() > deadline:
            return adj, edge_count
        
        # Get list of non-edges that can be added without creating triangles
        candidate_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in edge_set and not (adj[u] & (1 << v)):
                    if (adj[u] & adj[v]) == 0:
                        candidate_edges.append((u, v))
        
        random.shuffle(candidate_edges)
        
        for u, v in candidate_edges:
            if time.time() > deadline:
                break
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_count += 1
            if is_3_colorable(adj):
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edge_count -= 1
            else:
                edge_set.add((u, v))
        
        return adj, edge_count

    while time.time() < deadline:
        adj = [0] * n
        edge_count = 0
        
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
        
        if not is_3_colorable(adj):
            # Try to improve by adding more edges
            adj, edge_count = try_add_edges(adj, edge_count)
            
            if edge_count > best_count:
                best_count = edge_count
                best_edges = []
                for u in range(n):
                    mask = adj[u]
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        if u < v:
                            best_edges.append([u, v])
                        mask ^= lsb
                edge_set.clear()
                for u in range(n):
                    mask = adj[u]
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        if u < v:
                            edge_set.add((u, v))
                        mask ^= lsb
    
    if best_count == 0:
        clebsch_edges = []
        for u in range(n):
            for bit_idx in range(4):
                v = u ^ (1 << bit_idx)
                if u < v:
                    clebsch_edges.append([u, v])
            v = u ^ 15
            if u < v:
                clebsch_edges.append([u, v])
        best_edges = clebsch_edges
    
    return best_edges

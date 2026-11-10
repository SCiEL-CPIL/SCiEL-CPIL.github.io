import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_edge_count = 0
    best_edges = []
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Helper: check if adding edge (u, v) keeps graph K3,3-free
    # Optimized: only consider w, x that are both neighbors of v
    def can_add(u, v, adj_mask):
        if (adj_mask[u] >> v) & 1:
            return True
        
        mask_u = adj_mask[u]
        
        # Get all neighbors of v (excluding u and v themselves for the pair search)
        nv = adj_mask[v] & ~((1 << v)) & ~((1 << u))
        
        # Iterate over pairs (w, x) both in N(v), w < x
        # w and x must be neighbors of v, and we check if u,w,x have >=2 common neighbors
        # (after adding u-v, v becomes a common neighbor of u,w,x, so if they already have 2, we get 3)
        
        # Extract list of neighbors of v for iteration
        nv_list = []
        tmp = nv
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            nv_list.append(idx)
            tmp ^= lsb
        
        for i in range(len(nv_list)):
            w = nv_list[i]
            mask_w = adj_mask[w]
            for j in range(i + 1, len(nv_list)):
                x = nv_list[j]
                mask_x = adj_mask[x]
                common = mask_u & mask_w & mask_x
                if common.bit_count() >= 2:
                    return False
        return True

    def polish(adj_mask, edges):
        """Local search: try to add missing edges and do edge swaps."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding missing edges
            missing = [e for e in all_edges if e not in edges]
            random.shuffle(missing)
            for u, v in missing:
                if can_add(u, v, adj_mask):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    edges.add((u, v))
                    improved = True
            
            # Try edge swaps: remove one edge, add two others
            # This is expensive, so only do it if we have time
            if time.time() < deadline - 5:
                edge_list = list(edges)
                random.shuffle(edge_list)
                for e1 in edge_list:
                    if time.time() >= deadline - 2:
                        break
                    u1, v1 = e1
                    # Remove e1
                    adj_mask[u1] &= ~(1 << v1)
                    adj_mask[v1] &= ~(1 << u1)
                    edges.discard(e1)
                    
                    # Try to add two missing edges
                    missing = [e for e in all_edges if e not in edges]
                    random.shuffle(missing)
                    added = []
                    for e2 in missing:
                        u2, v2 = e2
                        if can_add(u2, v2, adj_mask):
                            adj_mask[u2] |= (1 << v2)
                            adj_mask[v2] |= (1 << u2)
                            edges.add(e2)
                            added.append(e2)
                        if len(added) == 2:
                            break
                    
                    if len(added) >= 1:
                        improved = True
                    else:
                        # Undo the removal
                        adj_mask[u1] |= (1 << v1)
                        adj_mask[v1] |= (1 << u1)
                        edges.add(e1)
                        
        return adj_mask, edges

    # Run multiple random greedy constructions
    while time.time() < deadline:
        # Initialize empty graph
        adj_mask = [0] * n
        edges = set()
        
        # Get a shuffled list of edges
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        # Greedy addition
        for u, v in shuffled_edges:
            if can_add(u, v, adj_mask):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges.add((u, v))
        
        # Add-back phase
        missing_edges = [e for e in all_edges if e not in edges]
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            if can_add(u, v, adj_mask):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges.add((u, v))
        
        # Polish with local search
        adj_mask, edges = polish(adj_mask, edges)
        
        # Record best
        if len(edges) > best_edge_count:
            best_edge_count = len(edges)
            best_edges = list(edges)
        
        # Check time
        if time.time() >= deadline:
            break
    
    # Final polish on the best graph if we have time
    if time.time() < deadline - 5 and best_edges:
        adj_mask = [0] * n
        for u, v in best_edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        edges = set(best_edges)
        adj_mask, edges = polish(adj_mask, edges)
        if len(edges) > best_edge_count:
            best_edge_count = len(edges)
            best_edges = list(edges)
    
    # Convert to list of lists
    result = [[u, v] for u, v in best_edges]
    return result

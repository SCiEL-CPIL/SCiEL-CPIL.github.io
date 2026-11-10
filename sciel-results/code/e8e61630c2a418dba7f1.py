import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def can_add(u, v, adj_mask):
        """
        Check if adding edge (u, v) keeps the graph K3,3-free.
        K3,3 exists if there are a1, a2 in N(v) such that
        |N(a1) & N(a2) & N(u)| >= 2.
        """
        if (adj_mask[u] >> v) & 1:
            return True
        
        N_u = adj_mask[u]
        N_v = adj_mask[v]
        
        # Iterate over pairs in N_v
        # Get list of neighbors of v
        # Optimization: precompute list of bits for N_v? 
        # n=20, so N_v has <= 19 bits.
        # Converting to list is fast.
        nv_list = []
        mask = N_v
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            nv_list.append(idx)
            mask ^= lsb
            
        len_nv = len(nv_list)
        for i in range(len_nv):
            a1 = nv_list[i]
            mask_a1 = adj_mask[a1]
            for j in range(i + 1, len_nv):
                a2 = nv_list[j]
                # Common neighbors of a1 and a2 that are also in N_u
                common = mask_a1 & adj_mask[a2] & N_u
                if common.bit_count() >= 2:
                    return False
        return True

    def polish(adj_mask, edges_set):
        improved = True
        no_improve_count = 0
        while improved and time.time() < deadline:
            improved = False
            no_improve_count += 1
            if no_improve_count > 3: # Faster exit from local optimum
                break
            
            # 1. Add edges
            missing = [e for e in all_edges if e not in edges_set]
            random.shuffle(missing)
            for u, v in missing:
                if can_add(u, v, adj_mask):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    edges_set.add((u, v))
                    improved = True
            
            # 2. Edge swaps
            if time.time() < deadline - 2:
                edge_list = list(edges_set)
                random.shuffle(edge_list)
                for e1 in edge_list:
                    if time.time() >= deadline - 1:
                        break
                    u1, v1 = e1
                    # Remove e1
                    adj_mask[u1] &= ~(1 << v1)
                    adj_mask[v1] &= ~(1 << u1)
                    edges_set.discard(e1)
                    
                    # Try to add a different edge
                    missing = [e for e in all_edges if e not in edges_set]
                    random.shuffle(missing)
                    added = None
                    for e2 in missing:
                        if can_add(e2[0], e2[1], adj_mask):
                            added = e2
                            break
                    
                    if added:
                        u2, v2 = added
                        adj_mask[u2] |= (1 << v2)
                        adj_mask[v2] |= (1 << u2)
                        edges_set.add(added)
                        improved = True
                    else:
                        # Undo removal
                        adj_mask[u1] |= (1 << v1)
                        adj_mask[v1] |= (1 << u1)
                        edges_set.add(e1)

        return adj_mask, edges_set

    best_edge_count = 0
    best_edges = []
    
    # Multi-start
    while time.time() < deadline:
        adj_mask = [0] * n
        edges_set = set()
        
        # Random greedy initialization
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if can_add(u, v, adj_mask):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges_set.add((u, v))
        
        adj_mask, edges_set = polish(adj_mask, edges_set)
        
        current_count = len(edges_set)
        if current_count > best_edge_count:
            best_edge_count = current_count
            best_edges = list(edges_set)
            
        # If we have a good result, we can try to polish it more or restart
        if time.time() < deadline - 5:
             # Occasionally re-polish the best to see if we can squeeze more
             if random.random() < 0.2 and best_edges:
                adj_mask_best = [0] * n
                edges_set_best = set(best_edges)
                for u, v in best_edges:
                    adj_mask_best[u] |= (1 << v)
                    adj_mask_best[v] |= (1 << u)
                
                adj_mask_best, edges_set_best = polish(adj_mask_best, edges_set_best)
                if len(edges_set_best) > best_edge_count:
                    best_edge_count = len(edges_set_best)
                    best_edges = list(edges_set_best)

    result = [[u, v] for u, v in best_edges]
    return result

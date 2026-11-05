import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def time_left():
        return deadline - time.time()

    # Fast verifier for a specific subset including a new vertex
    def check_violation_with_new_vertex(adj_masks, k, new_mask, subset_5):
        """
        adj_masks: list of masks for vertices 0..k-1
        k: index of new vertex
        new_mask: mask of edges from k to 0..k-1
        subset_5: tuple of 5 vertices from 0..k-1
        Returns True if the induced subgraph on subset_5 U {k} is regular.
        """
        mask_5 = 0
        for v in subset_5:
            mask_5 |= (1 << v)
        
        # Degree of k in this subgraph
        d_k = (new_mask & mask_5).bit_count()
        
        # Check if all vertices in subset_5 have degree d_k
        for v in subset_5:
            # Degree of v in subset_5 U {k}
            # Edges within subset_5: adj_masks[v] & mask_5
            # Edge to k: 1 if v in new_mask
            deg_v = (adj_masks[v] & mask_5).bit_count()
            if new_mask & (1 << v):
                deg_v += 1
            
            if deg_v != d_k:
                return False
        
        return True

    def is_valid_full_graph(adj_masks, n):
        """
        Full verification for safety, used at the end.
        """
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                degs = []
                for v in subset:
                    degs.append((adj_masks[v] & mask).bit_count())
                if len(set(degs)) == 1:
                    return False
        return True

    def greedy_build(target_n, max_restarts=500):
        """
        Builds a graph of size target_n using greedy incremental addition.
        """
        for restart in range(max_restarts):
            if time_left() < 5.0:
                return None
            
            adj = [0] * target_n
            valid_build = True
            
            for k in range(target_n):
                if k < 5:
                    # No size-6 subgraphs possible
                    for i in range(k):
                        if random.random() < 0.5:
                            adj[k] |= (1 << i)
                            adj[i] |= (1 << k)
                    continue
                
                # Try to find a valid mask for vertex k
                found = False
                attempts = 0
                max_attempts = 200
                
                while not found and attempts < max_attempts:
                    attempts += 1
                    # Generate random mask
                    mask = 0
                    for i in range(k):
                        if random.random() < 0.5:
                            mask |= (1 << i)
                    
                    # Check all subsets of size 5 from previous vertices
                    # C(k, 5) subsets
                    # Optimization: Precompute list of 5-combinations?
                    # For k up to 20, C(20,5) = 15504. This is okay.
                    valid_mask = True
                    for subset_5 in itertools.combinations(range(k), 5):
                        if check_violation_with_new_vertex(adj, k, mask, subset_5):
                            valid_mask = False
                            break
                    
                    if valid_mask:
                        # Commit
                        for i in range(k):
                            if mask & (1 << i):
                                adj[k] |= (1 << i)
                                adj[i] |= (1 << k)
                        found = True
                        break
                
                if not found:
                    valid_build = False
                    break
            
            if valid_build:
                # Final full check to be safe
                if is_valid_full_graph(adj, target_n):
                    return adj
        
        return None

    # Try to build n=21 first, then 20
    # Note: The task says 20-vertex graphs are known. 21 is the goal.
    
    best_adj = None
    best_n = 0
    
    # Try n=21
    if time_left() > 10.0:
        adj = greedy_build(21, max_restarts=100)
        if adj:
            best_adj = adj
            best_n = 21
    
    # If 21 failed, try n=20
    if best_n < 20 and time_left() > 10.0:
        adj = greedy_build(20, max_restarts=500)
        if adj:
            best_adj = adj
            best_n = 20
    
    # Fallback to n=19 (Paley) if 20 fails
    if best_n < 19:
        # Build Paley(19)
        p = 19
        adj = [0] * p
        for i in range(p):
            for j in range(i+1, p):
                if pow((i-j) % p, (p-1)//2, p) == 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        if is_valid_full_graph(adj, 19):
            best_adj = adj
            best_n = 19
    
    if best_adj:
        edges = []
        for i in range(best_n):
            m = best_adj[i]
            for j in range(i+1, best_n):
                if m & (1 << j):
                    edges.append([i, j])
        return {'n': best_n, 'edges': edges}
    
    # Last resort: n=5
    return {'n': 5, 'edges': []}

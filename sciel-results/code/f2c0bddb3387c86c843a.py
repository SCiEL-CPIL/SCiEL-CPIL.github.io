import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 20
    range_n = range(n)
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_all_edges = len(all_edges)
    
    # Precompute for each edge (u,v), the list of other vertices w != u,v
    # and for each vertex pair, which edges are "conflicting"
    
    # Key insight: A graph is K3,3-free iff for every pair of vertices (u,v),
    # they have at most 2 common neighbors.
    # 
    # When we add edge (u,v), it can only violate the constraint for pairs
    # that involve u or v. Specifically:
    # - For any w != u,v: if v is in N(w), then adding (u,v) increases 
    #   common neighbors of (u,w) by 1. Similarly for (v,w) if u is in N(w).
    #
    # So to check if adding (u,v) is valid, we need to check:
    # For each w where v is adjacent to w: count_common(u,w) < 3
    # For each w where u is adjacent to w: count_common(v,w) < 3
    
    best_edges = []
    best_count = 0
    
    def greedy_build_with_order(order):
        """Build a K3,3-free graph using greedy edge addition with given order."""
        adj = [0] * n
        edges = []
        
        for u, v in order:
            # Check if adding (u,v) keeps graph K3,3-free
            # Need to check pairs (u,w) for all w adjacent to v (w != u)
            # and pairs (v,w) for all w adjacent to u (w != v)
            
            valid = True
            
            # Check pairs (u, w) where w is a neighbor of v (w != u)
            # After adding (u,v), common neighbors of u and w = 
            #   (N(u) & N(w)) | {v if v in N(w)}
            # Since we're adding (u,v), v becomes a neighbor of u.
            # But v was already a neighbor of w (that's why we're checking).
            # So new common count = old common count + 1 (for v)
            
            # Actually let's think more carefully:
            # Before adding (u,v): N(u) and N(v) don't include each other
            # After adding (u,v): N(u) gains v, N(v) gains u
            # 
            # For pair (u, w) where w != u, v:
            #   Old common: |N(u) ∩ N(w)|
            #   New common: |N(u) ∩ N(w)| + (1 if v ∈ N(w) else 0)
            #   because v is now in N(u), and v may or may not be in N(w)
            #
            # For pair (v, w) where w != u, v:
            #   Old common: |N(v) ∩ N(w)|
            #   New common: |N(v) ∩ N(w)| + (1 if u ∈ N(w) else 0)
            
            # Check all w that are neighbors of v (these are the w where v ∈ N(w))
            mask_v = adj[v]
            # Remove u from consideration if present (shouldn't be, but just in case)
            for w in range_n:
                if w == u or w == v:
                    continue
                if (mask_v >> w) & 1:
                    # v is in N(w), so adding (u,v) adds v to common neighbors of (u,w)
                    common_uw = (adj[u] & adj[w]).bit_count()
                    if common_uw >= 2:  # would become 3
                        valid = False
                        break
            
            if not valid:
                continue
            
            mask_u = adj[u]
            for w in range_n:
                if w == u or w == v:
                    continue
                if (mask_u >> w) & 1:
                    # u is in N(w), so adding (u,v) adds u to common neighbors of (v,w)
                    common_vw = (adj[v] & adj[w]).bit_count()
                    if common_vw >= 2:  # would become 3
                        valid = False
                        break
            
            if valid:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        
        return edges, adj
    
    def local_search(edges, adj):
        """Improve a K3,3-free graph via local search."""
        current_edges = edges[:]
        current_adj = adj[:]
        current_count = len(current_edges)
        edge_set = set((min(e), max(e)) for e in current_edges)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Try adding edges
            for u, v in all_edges:
                if time.time() > deadline - 10:
                    break
                if (u, v) in edge_set:
                    continue
                
                # Check if adding (u,v) is valid
                mask_v = current_adj[v]
                valid = True
                for w in range_n:
                    if w == u or w == v:
                        continue
                    if (mask_v >> w) & 1:
                        common_uw = (current_adj[u] & current_adj[w]).bit_count()
                        if common_uw >= 2:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                mask_u = current_adj[u]
                for w in range_n:
                    if w == u or w == v:
                        continue
                    if (mask_u >> w) & 1:
                        common_vw = (current_adj[v] & current_adj[w]).bit_count()
                        if common_vw >= 2:
                            valid = False
                            break
                
                if valid:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.append([u, v])
                    edge_set.add((u, v))
                    current_count += 1
                    improved = True
        
        return current_edges, current_adj, current_count
    
    # Phase 1: Multiple greedy builds with different random orders
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Create random order
        order = all_edges[:]
        random.shuffle(order)
        
        edges, adj = greedy_build_with_order(order)
        edges, adj, count = local_search(edges, adj)
        
        if count > best_count:
            best_count = count
            best_edges = edges[:]
    
    # Phase 2: Try to improve the best solution with more aggressive local search
    if best_edges and time.time() < deadline - 10:
        # Reconstruct adj from best_edges
        adj = [0] * n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        best_edges, adj, best_count = local_search(best_edges, adj)
    
    return best_edges

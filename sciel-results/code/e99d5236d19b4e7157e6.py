"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We stop at 145s.
    deadline = start_time + 145.0
    
    n = 20
    
    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u, v) maintains K2,3-freeness.
        """
        # 1. Check common neighbors of u and v.
        # The set N(u) ∩ N(v) does not change when adding (u,v) to the graph 
        # in terms of *other* vertices w. u is not in N(u), v is not in N(v).
        # So we just need current count <= 2.
        common_uv = np.sum(adj[u] & adj[v])
        if common_uv > 2:
            return False
            
        # 2. Check pairs (u, w) for w in N(v).
        # Adding (u,v) adds v to N(u).
        # For any w in N(v) (w != u), v is now a common neighbor of u and w.
        # So new |N(u) ∩ N(w)| = old |N(u) ∩ N(w)| + 1.
        # We need new <= 2, so old <= 1.
        neighbors_v = np.where(adj[v])[0]
        for w in neighbors_v:
            if w == u:
                continue
            # Check common neighbors of u and w
            # If this is > 1, adding (u,v) makes it > 2, which is invalid.
            common_uw = np.sum(adj[u] & adj[w])
            if common_uw > 1:
                return False
                
        # 3. Check pairs (v, w) for w in N(u).
        # Adding (u,v) adds u to N(v).
        # For any w in N(u) (w != v), u is now a common neighbor of v and w.
        # So new |N(v) ∩ N(w)| = old |N(v) ∩ N(w)| + 1.
        # We need new <= 2, so old <= 1.
        neighbors_u = np.where(adj[u])[0]
        for w in neighbors_u:
            if w == v:
                continue
            # Check common neighbors of v and w
            common_vw = np.sum(adj[v] & adj[w])
            if common_vw > 1:
                return False
                
        return True

    def greedy_spectral(adj, time_limit):
        """
        Deterministic greedy construction.
        We compute a 'score' for each potential edge based on a proxy for 
        spectral centrality or simply use a fixed deterministic ordering 
        (e.g., by vertex index sum, then diff) to ensure reproducibility 
        and avoid the variance of random tie-breaking.
        
        However, to be 'spectral', we can approximate the benefit of an edge 
        by its contribution to the trace of A^2 or similar, but for K2,3-free,
        minimizing common neighbors is the primary constraint.
        
        Let's use a deterministic ordering: 
        1. Sort all edges by (min_common_neighbors, u, v).
        2. Iterate and add if valid.
        """
        edges = []
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # We will re-evaluate the "best" edge at each step to ensure we always
        # pick the one with the lowest current common neighbors, breaking ties
        # deterministically by (u, v) index.
        
        remaining_edges = set(all_edges)
        
        while remaining_edges:
            if time.time() > time_limit:
                break
                
            best_edge = None
            min_cost = float('inf')
            best_key = (n, n) # Max possible key for tie-breaking
            
            for u, v in remaining_edges:
                if is_valid_add(adj, u, v):
                    cost = np.sum(adj[u] & adj[v])
                    # Tie-breaker: smaller u, then smaller v
                    key = (cost, u, v)
                    if key < (min_cost, best_key[0], best_key[1]) if best_edge else True:
                         # Logic fix: compare tuple directly
                         if best_edge is None or (cost, u, v) < (min_cost, best_edge[0], best_edge[1]):
                             min_cost = cost
                             best_edge = (u, v)
                             best_key = (u, v)
            
            if best_edge is None:
                break
                
            u, v = best_edge
            adj[u, v] = True
            adj[v, u] = True
            edges.append([u, v])
            remaining_edges.discard((u, v))
            
        return edges, adj

    def local_search_1swap(adj, edges, time_limit):
        """
        Deterministic Local Search.
        1. Try to add any valid missing edge.
        2. If stuck, try removing one edge and then greedily filling.
        We do this systematically for all edges to ensure we find the local optimum.
        """
        best_edges = edges
        best_adj = adj
        best_count = len(best_edges)
        
        current_adj = best_adj.copy()
        current_edges = list(best_edges)
        
        # Phase 1: Greedy Fill (Deterministic order)
        all_possible = [(i, j) for i in range(n) for j in range(i + 1, n) if not current_adj[i, j]]
        # Sort to be deterministic
        all_possible.sort()
        for u, v in all_possible:
            if time.time() > time_limit:
                break
            if is_valid_add(current_adj, u, v):
                current_adj[u, v] = True
                current_adj[v, u] = True
                current_edges.append([u, v])
                
        if len(current_edges) > best_count:
            best_edges = current_edges
            best_adj = current_adj.copy()
            best_count = len(best_edges)
            
        # Phase 2: Systematic 1-Edge Swap
        # For each edge in the current graph, try removing it and then greedily filling.
        # If the resulting graph has more edges, keep it.
        
        # We iterate over a snapshot of the current edges to avoid index issues
        # if we update best.
        # To be efficient, we only do a few passes or limit iterations.
        
        max_passes = 5
        for _ in range(max_passes):
            if time.time() > time_limit:
                break
            
            improved = False
            # Iterate over edges in the *current* best graph
            # We need to be careful: if we remove an edge and refill, we might add a different set.
            # We will try removing each edge from the *best* graph found so far.
            
            best_edges_snapshot = list(best_edges)
            
            for i, (u, v) in enumerate(best_edges_snapshot):
                if time.time() > time_limit:
                    break
                
                # Create a copy of the best graph
                temp_adj = best_adj.copy()
                temp_adj[u, v] = False
                temp_adj[v, u] = False
                
                # Try to greedily fill the temp graph
                # We want to see if we can get MORE edges than best_count
                # Note: We start with best_count - 1 edges.
                # We need to find a set of edges to add such that total > best_count.
                # This is equivalent to finding a valid edge set to add to temp_adj 
                # that has size > 1 (since we removed 1, we need to add at least 2 to improve,
                # or 1 to tie, but we want improvement).
                
                # Actually, simpler: just run a greedy fill on temp_adj and see the result.
                
                # Get missing edges
                missing = [(x, y) for x in range(n) for y in range(x + 1, n) if not temp_adj[x, y]]
                missing.sort()
                
                # Greedy fill on temp_adj
                for x, y in missing:
                    if time.time() > time_limit:
                        break
                    if is_valid_add(temp_adj, x, y):
                        temp_adj[x, y] = True
                        temp_adj[y, x] = True
                        
                temp_edge_count = np.sum(temp_adj) // 2
                
                if temp_edge_count > best_count:
                    # Update best
                    best_adj = temp_adj
                    # Reconstruct edge list
                    best_edges = [[i, j] for i in range(n) for j in range(i + 1, n) if temp_adj[i, j]]
                    best_count = temp_edge_count
                    improved = True
                    break # Restart the loop with the new best graph? Or continue?
                    # Breaking and restarting the outer loop ensures we process the new best graph.
            
            if not improved:
                break
                
        return best_edges, best_adj

    # Main Logic
    adj = np.zeros((n, n), dtype=bool)
    edges, adj = greedy_spectral(adj, deadline - 5)
    
    # Apply local search
    improved_edges, improved_adj = local_search_1swap(adj, edges, deadline - 2)
    
    return improved_edges

# EVOLVE-BLOCK-END

import numpy as np
import time
import random

def construct():
    n = 16
    target_edges = 65
    deadline = time.time() + 110  # Safety margin
    
    # Precompute all possible edges and non-edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    # all_pairs has 120 elements
    
    # Precompute all triangles (triples)
    triples = [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
    # triples has 560 elements
    
    best_score = float('inf')
    best_adj = None
    best_edges = None
    
    def get_score(adj):
        # adj is a list of 16 ints (bitmasks)
        # degs is a list of 16 ints
        degs = [bin(adj[i]).count('1') for i in range(n)]
        max_sum = 0
        for u, v, w in triples:
            # Check if triangle exists
            if (adj[u] >> v) & 1 and (adj[v] >> w) & 1 and (adj[u] >> w) & 1:
                s = degs[u] + degs[v] + degs[w]
                if s > max_sum:
                    max_sum = s
        return max_sum

    def random_graph():
        # Pick 65 random edges
        edges = random.sample(all_pairs, target_edges)
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj, edges

    def local_search(adj, edges, current_score):
        # edges is a list of tuples
        edge_set = set(edges)
        non_edges = [p for p in all_pairs if p not in edge_set]
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            best_swap = None
            best_swap_score = current_score
            best_swap_edges = None
            
            # Iterate all possible swaps
            for e_remove in edges:
                u, v = e_remove
                for e_add in non_edges:
                    x, y = e_add
                    # Ensure we don't add the same edge
                    if (u, v) == (x, y) or (u, v) == (y, x):
                        continue
                    # Check if (x,y) is already an edge (it shouldn't be, from non_edges)
                    
                    # Apply swap to a copy
                    new_adj = adj[:]
                    # Remove u-v
                    new_adj[u] &= ~(1 << v)
                    new_adj[v] &= ~(1 << u)
                    # Add x-y
                    new_adj[x] |= (1 << y)
                    new_adj[y] |= (1 << x)
                    
                    new_score = get_score(new_adj)
                    
                    if new_score < best_swap_score:
                        best_swap_score = new_score
                        best_swap = (e_remove, e_add)
                        best_swap_edges = edges[:]
                        # Update edges list for the potential swap
                        best_swap_edges.remove(e_remove)
                        best_swap_edges.append(e_add)
                        # We don't need to break here, we want the best swap
                        # But if we find a 0 score (no triangles), we can break?
                        # Score 0 means no triangles. But we have 65 edges, likely has triangles.
                        # Actually, min score is not 0.
            
            if best_swap:
                # Apply the best swap
                e_remove, e_add = best_swap
                adj = new_adj # Wait, new_adj is from the last iteration?
                # We need to re-apply or store the adj from the best swap
                # Let's re-apply from the stored edges or just use the new_adj from the best iteration
                # It's safer to store the adj corresponding to best_swap_score
                
                # Re-do the application for the best swap found
                # Actually, we can just use the new_adj from the iteration that produced best_swap_score
                # But `new_adj` is overwritten in the loop.
                # Let's store `best_new_adj`.
                
                # Optimization: Store best_new_adj in the loop
                # Let's refactor the loop to store best_new_adj
                
                # For now, assume we have the best_adj_candidate
                # We need to re-implement the loop to keep track of the best_adj
                pass
            
        return adj, edges, current_score

    # Restart local search with proper state tracking
    def local_search_v2(adj, edges, current_score):
        edge_set = set(edges)
        non_edges = [p for p in all_pairs if p not in edge_set]
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            best_swap_score = current_score
            best_new_adj = None
            best_new_edges = None
            best_e_remove = None
            best_e_add = None
            
            for e_remove in edges:
                u, v = e_remove
                # Optimization: If deg(u) + deg(v) is low, removing this edge might not help much?
                # But it reduces degrees.
                
                for e_add in non_edges:
                    x, y = e_add
                    
                    new_adj = adj[:]
                    new_adj[u] &= ~(1 << v)
                    new_adj[v] &= ~(1 << u)
                    new_adj[x] |= (1 << y)
                    new_adj[y] |= (1 << x)
                    
                    new_score = get_score(new_adj)
                    
                    if new_score < best_swap_score:
                        best_swap_score = new_score
                        best_new_adj = new_adj
                        best_new_edges = edges[:]
                        best_new_edges.remove(e_remove)
                        best_new_edges.append(e_add)
                        best_e_remove = e_remove
                        best_e_add = e_add
            
            if best_new_adj is not None:
                adj = best_new_adj
                edges = best_new_edges
                current_score = best_swap_score
                improved = True
                # Update non_edges
                edge_set = set(edges)
                non_edges = [p for p in all_pairs if p not in edge_set]
            
        return adj, edges, current_score

    # Main loop
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        adj, edges = random_graph()
        score = get_score(adj)
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            best_edges = edges[:]
            
        # Run local search
        adj, edges, score = local_search_v2(adj, edges, score)
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            best_edges = edges[:]
            
        # Optional: Try a few more random restarts if time permits
        # The local search might get stuck.
        
    # Return the best edges
    # Convert adj back to edges if needed, but we have best_edges
    return {'n': n, 'edges': best_edges}

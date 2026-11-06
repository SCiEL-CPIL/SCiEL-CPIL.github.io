import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n = 16
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)
    
    # Bitmasks for adjacency
    # adj_mask[i] is a bitmask of neighbors of vertex i
    adj_mask = [0] * n
    
    # For triangle-free constraint: if we add edge (u, v), then N(u) and N(v) must be disjoint.
    # We can precompute for each potential edge (u, v) the set of edges that would create a triangle.
    # Actually, simpler: during backtracking, we maintain adj_mask.
    # When considering adding edge (u, v), check if adj_mask[u] & adj_mask[v] == 0.
    
    # For 3-colorability:
    # We can use a fast backtracking solver for 3-coloring.
    # Or, since n=16 is small, we can integrate the check into the main search or do it at leaves.
    # Given the density, checking at every node might be slow.
    # However, the space of triangle-free graphs is much smaller than all graphs.
    
    # Let's use a hybrid: Backtrack to build triangle-free graphs, and check 3-colorability
    # only when we reach a maximal triangle-free graph or periodically.
    # But "maximal" is hard to define in backtracking.
    # Better: Backtrack through all potential edges.
    # To maximize edges, we should prioritize adding edges.
    
    # Strategy:
    # 1. Sort edges by a heuristic that favors high-degree vertices or central structures?
    #    Actually, for maximizing edges in triangle-free graphs, the structure is often bipartite-like
    #    plus some odd cycles (like Grötzsch).
    #    Random order with random restarts is okay, but exact search is better.
    
    # Let's implement a depth-first search over the edge set.
    # State: index of next edge to consider, current adj_mask, current edge count.
    # Pruning:
    #   - If time > deadline, return best found.
    #   - If adding edge (u, v) creates a triangle, skip it.
    #   - We don't strictly need to check 3-colorability at every step, but we must ensure the final graph is NOT 3-colorable.
    #   - If the current graph is already 3-colorable, adding edges might make it 4-chromatic.
    #   - If the current graph is NOT 3-colorable, any supergraph is also NOT 3-colorable.
    #     So if we find a non-3-colorable subgraph, we can greedily add all remaining valid edges to maximize count!
    
    best_edges = []
    best_count = 0
    global_deadline = deadline
    
    # Fast 3-colorability checker using backtracking with bitmasks
    def is_3_colorable(adj_mask_list):
        # adj_mask_list is a list of integers (bitmasks)
        # Try to color vertices 0..15 with 3 colors
        # color[i] = 0, 1, 2 or -1
        color = [-1] * n
        
        # Order vertices by degree descending for efficiency
        order = sorted(range(n), key=lambda i: bin(adj_mask_list[i]).count('1'), reverse=True)
        
        def dfs(idx):
            if idx == n:
                return True
            u = order[idx]
            # Determine available colors
            # A color c is invalid if any neighbor has color c
            # We can compute a bitmask of used colors by neighbors
            used_colors = 0
            # Iterate neighbors of u
            mask = adj_mask_list[u]
            # For each neighbor v, if color[v] is set, mark that color as used
            # This loop is O(n) in worst case, but n=16.
            # We can precompute for each vertex u, the list of neighbors.
            # But adj_mask_list changes.
            
            # Let's just check against current colors
            for v in range(n):
                if (mask >> v) & 1:
                    if color[v] != -1:
                        used_colors |= (1 << color[v])
            
            for c in range(3):
                if not (used_colors & (1 << c)):
                    color[u] = c
                    if dfs(idx + 1):
                        return True
                    color[u] = -1
            return False
        
        return dfs(0)
    
    def solve():
        nonlocal best_edges, best_count, global_deadline
        
        # We will search over the edges.
        # To improve performance, we can shuffle the edge order for different restarts.
        # However, exact backtracking over 120 edges is 2^120, which is impossible.
        # We MUST prune heavily.
        
        # Key Insight: If a graph is not 3-colorable, we stop searching for that branch
        # and just greedily fill all remaining triangle-free edges.
        
        # So the search is: Find the first non-3-colorable triangle-free graph.
        # To maximize edges, we want to find such a graph with as many edges as possible?
        # No, we want the *final* graph to have max edges.
        # If we find a core that is 4-chromatic, we add all compatible edges.
        # So we should search for 4-chromatic cores.
        
        # Heuristic for edge ordering:
        # Edges that are "critical" for 4-chromaticity.
        # Grötzsch graph is a good starting point.
        
        # Let's try a different approach:
        # 1. Generate a set of candidate 4-chromatic triangle-free graphs (cores).
        # 2. For each core, greedily add all other edges that don't form triangles.
        # 3. Keep the one with max edges.
        
        # How to generate cores?
        # Use backtracking to find minimal 4-chromatic triangle-free graphs?
        # Or just use random restarts of the backtracking search, but stop as soon as
        # the current graph becomes 4-chromatic, then expand.
        
        # Let's implement:
        # def backtrack(edge_idx, adj_mask, edge_count, current_edges):
        #   if time > deadline: return
        #   if edge_idx == num_edges:
        #       if not is_3_colorable(adj_mask):
        #           # Maximize this graph
        #           expanded = expand(adj_mask, current_edges)
        #           if len(expanded) > best_count:
        #               best_count = len(expanded)
        #               best_edges = expanded
        #       return
        #   
        #   u, v = all_edges[edge_idx]
        #   
        #   # Option 1: Skip edge (u, v)
        #   backtrack(edge_idx + 1, adj_mask, edge_count, current_edges)
        #   
        #   # Option 2: Add edge (u, v) if triangle-free
        #   if (adj_mask[u] & adj_mask[v]) == 0:
        #       new_adj = adj_mask.copy()
        #       new_adj[u] |= (1 << v)
        #       new_adj[v] |= (1 << u)
        #       
        #       # Check if this new graph is 4-chromatic?
        #       # If yes, we can expand and update best, then prune this branch (no need to add more edges to a 4-chromatic graph to find a "core", 
        #       # BUT adding more edges might allow OTHER edges to be added? No, triangle-free constraint is local.
        #       # If G is 4-chromatic, G' (supergraph) is 4-chromatic.
        #       # So if we find a 4-chromatic subgraph, the max edges for this branch is obtained by adding ALL possible remaining edges.
        #       # So we can prune the rest of the search tree for this branch.
        #       
        #       if not is_3_colorable(new_adj):
        #           # Expand greedily
        #           expanded = expand_greedy(new_adj, current_edges + [(u, v)])
        #           if len(expanded) > best_count:
        #               best_count = len(expanded)
        #               best_edges = expanded
        #           return # Prune
        #       
        #       backtrack(edge_idx + 1, new_adj, edge_count + 1, current_edges + [(u, v)])
        
        # Problem: is_3_colorable is called frequently.
        # n=16 is small, so it should be fast enough if we optimize.
        
        def expand_greedy(adj_mask, current_edges):
            # Add all edges that don't form triangles
            # current_edges is a list of (u, v)
            # adj_mask is the state after current_edges
            # We iterate through all_edges not in current_edges
            # If compatible, add it.
            
            # To make it deterministic and fast, we just check all remaining edges.
            # Note: Adding an edge might block others? No, triangle-free constraint is monotonic.
            # If (u, v) is compatible with current adj, adding it only removes potential edges
            # that would form triangles with u or v.
            # So we can just iterate and add if possible.
            
            edges_set = set(current_edges)
            new_adj = adj_mask[:]
            count = len(current_edges)
            
            # Iterate through all edges
            # To be safe, we might need to iterate multiple times?
            # No, if an edge is compatible now, it remains compatible?
            # Wait. If I add edge (a, b), it might make edge (c, d) incompatible?
            # (c, d) is incompatible if c is neighbor of d in new graph.
            # Adding (a, b) makes a neighbor of b and b neighbor of a.
            # So (c, d) could become incompatible if c=a, d=b (already added) or
            # if c is a and d is a neighbor of b? No.
            # Triangle formed by (a, b) and (x, y) exists if x is neighbor of y?
            # No, triangle is (a, b, c) if a-b, b-c, c-a.
            # If we add (a, b), we forbid any edge (c, d) such that c=a, d in N(b) OR d=a, c in N(b) etc.
            # Actually, adding (a, b) forbids edges between N(a) and N(b).
            # So if I add (a, b), some future edges (c, d) where c in N(a) and d in N(b) become forbidden.
            # So the order of addition matters for which edges we CAN add?
            # No, the set of edges we CAN add to a graph G is fixed.
            # The maximal triangle-free supergraph of G is unique?
            # No. Consider a 4-cycle 1-2-3-4-1.
            # We can add (1,3) or (2,4) but not both.
            # So the maximal supergraph depends on choices.
            # Greedy addition might not yield the maximum number of edges.
            # However, for the purpose of this task, we just need a valid graph with high edge count.
            # The previous program used greedy.
            # To improve, we could use a better expansion strategy, e.g., try to add edges that block the fewest other edges.
            
            # For now, let's use a simple greedy with a heuristic:
            # Sort remaining edges by "degree of freedom" or just random?
            # Let's try adding edges in an order that minimizes conflicts.
            
            remaining = [e for e in all_edges if e not in edges_set]
            # Heuristic: Add edges between vertices with low degree first? Or high?
            # In triangle-free graphs, high degree vertices are constrained.
            # Let's just shuffle and add.
            
            import random
            random.shuffle(remaining)
            
            for u, v in remaining:
                if (new_adj[u] & new_adj[v]) == 0:
                    new_adj[u] |= (1 << v)
                    new_adj[v] |= (1 << u)
                    edges_set.add((u, v))
                    count += 1
            
            return list(edges_set), count
        
        # To avoid deep recursion issues and allow time checking, use iterative or careful recursion.
        # Recursion depth is 120, which is fine.
        
        # We will run multiple restarts with different edge orderings.
        
        for trial in range(100):
            if time.time() > global_deadline - 5:
                break
            
            # Shuffle edges to explore different parts of the search space
            shuffled_edges = all_edges[:]
            import random
            random.shuffle(shuffled_edges)
            
            # Reset state
            adj_mask = [0] * n
            current_edges = []
            
            # Backtracking search
            # We want to find a 4-chromatic subgraph quickly.
            # If we find one, we expand it.
            
            # Stack: (edge_idx, adj_mask, current_edges)
            # To save memory, we can use recursion.
            
            def backtrack(idx):
                nonlocal best_edges, best_count, global_deadline
                
                if time.time() > global_deadline - 2:
                    return False # Signal to stop
                
                if idx == num_edges:
                    # Check if current graph is 4-chromatic
                    if not is_3_colorable(adj_mask):
                        expanded, count = expand_greedy(adj_mask, current_edges)
                        if count > best_count:
                            best_count = count
                            best_edges = expanded
                    return False
                
                u, v = shuffled_edges[idx]
                
                # Option 1: Skip
                if backtrack(idx + 1):
                    return True
                
                # Option 2: Add
                if (adj_mask[u] & adj_mask[v]) == 0:
                    # Save state
                    old_u = adj_mask[u]
                    old_v = adj_mask[v]
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    current_edges.append((u, v))
                    
                    # Check if 4-chromatic
                    if not is_3_colorable(adj_mask):
                        expanded, count = expand_greedy(adj_mask, current_edges)
                        if count > best_count:
                            best_count = count
                            best_edges = expanded
                        # Restore state
                        adj_mask[u] = old_u
                        adj_mask[v] = old_v
                        current_edges.pop()
                        return False # No need to explore deeper, this branch is "done" for optimization
                    else:
                        if backtrack(idx + 1):
                            adj_mask[u] = old_u
                            adj_mask[v] = old_v
                            current_edges.pop()
                            return True
                    
                    # Restore state
                    adj_mask[u] = old_u
                    adj_mask[v] = old_v
                    current_edges.pop()
                
                return False
            
            backtrack(0)
            
            # If we found a good solution, we might break early?
            # No, keep searching for better ones.
            
            # Optional: If best_count is close to theoretical max, stop?
            # Max edges in triangle-free graph on 16 vertices is floor(16^2/4) = 64.
            # If best_count == 64, break.
            if best_count >= 64:
                break

    solve()
    
    if best_edges:
        # Ensure the format is list of lists [u, v]
        return [list(e) for e in best_edges]
    
    # Fallback: Grötzsch graph
    def groetzch_graph():
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        edges = [list(e) for e in cycle]
        for u, w in cycle:
            edges += [[u, 5 + w], [w, 5 + u]]
        edges += [[5 + u, 10] for u in range(5)]
        return edges

    return groetzch_graph()

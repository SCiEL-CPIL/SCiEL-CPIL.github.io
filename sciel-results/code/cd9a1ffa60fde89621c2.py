"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    """
    Construct a triangle-free graph with no independent set of size 10,
    maximizing n. Uses a branch-and-bound vertex extension approach.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Start with the Andrasfai graph And(9) on 26 vertices
    # Z_26 with i ~ j iff (j - i) mod 26 is 1 mod 3
    n_base = 26
    adj = [[False] * n_base for _ in range(n_base)]
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if (j - i) % n_base % 3 == 1:
                adj[i][j] = True
                adj[j][i] = True
    
    def has_triangle(adj, n):
        """Check if graph has any triangle."""
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    for k in range(j + 1, n):
                        if adj[i][k] and adj[j][k]:
                            return True
        return False
    
    def max_independent_set_size(adj, n):
        """Find the maximum independent set size using branch and bound."""
        # Use a greedy upper bound: for each vertex, count non-neighbors
        # and use branch and bound
        
        best = [0]
        
        def branch_and_bound(vertex_set, size):
            if size > best[0]:
                best[0] = size
            
            if not vertex_set:
                return
            
            # Upper bound: size + len(vertex_set)
            if size + len(vertex_set) <= best[0]:
                return
            
            # Choose vertex with maximum degree in the induced subgraph
            v = vertex_set[0]
            remaining = vertex_set[1:]
            
            # Branch 1: include v
            non_neighbors = [u for u in remaining if not adj[v][u]]
            branch_and_bound(non_neighbors, size + 1)
            
            # Branch 2: exclude v
            branch_and_bound(remaining, size)
        
        all_vertices = list(range(n))
        branch_and_bound(all_vertices, 0)
        return best[0]
    
    def is_valid(adj, n):
        """Check if graph is triangle-free and has no independent set of size 10."""
        if has_triangle(adj, n):
            return False
        if max_independent_set_size(adj, n) >= 10:
            return False
        return True
    
    # Try to extend the graph by adding vertices
    best_n = n_base
    best_adj = [row[:] for row in adj]
    
    current_n = n_base
    current_adj = [row[:] for row in adj]
    
    # Try to add vertices one by one
    while current_n < 40 and time.time() < deadline:
        new_n = current_n + 1
        new_adj = [row[:] + [False] for row in current_adj]
        new_adj.append([False] * new_n)
        
        # Try different strategies for connecting the new vertex
        best_strategy = None
        best_indep = 10  # We want to keep independent set < 10
        
        # Strategy: connect to a subset of existing vertices
        # The new vertex v_new should not create a triangle
        # and should not be part of an independent set of size 10
        
        # To avoid triangles: if v_new is connected to u, then for any w connected to u,
        # v_new cannot be connected to w.
        
        # Let's try connecting to vertices with different degrees
        # and check validity
        
        candidates = []
        for mask in range(1, min(2 ** current_n, 1000)):
            # Try this connection pattern
            if time.time() > deadline - 10:
                break
            
            trial_adj = [row[:] for row in new_adj]
            for i in range(current_n):
                if mask & (1 << i):
                    trial_adj[i][new_n - 1] = True
                    trial_adj[new_n - 1][i] = True
            
            if has_triangle(trial_adj, new_n):
                continue
            
            indep_size = max_independent_set_size(trial_adj, new_n)
            if indep_size < 10:
                candidates.append((indep_size, mask, trial_adj))
        
        if candidates:
            # Pick the one with smallest independent set size
            candidates.sort(key=lambda x: x[0])
            best_indep, best_mask, best_trial = candidates[0]
            current_n = new_n
            current_adj = best_trial
            if current_n > best_n:
                best_n = current_n
                best_adj = [row[:] for row in current_adj]
        else:
            # Try with random masks
            found = False
            for _ in range(100):
                if time.time() > deadline - 10:
                    break
                mask = random.getrandbits(current_n)
                if mask == 0:
                    continue
                
                trial_adj = [row[:] for row in new_adj]
                for i in range(current_n):
                    if mask & (1 << i):
                        trial_adj[i][new_n - 1] = True
                        trial_adj[new_n - 1][i] = True
                
                if has_triangle(trial_adj, new_n):
                    continue
                
                indep_size = max_independent_set_size(trial_adj, new_n)
                if indep_size < 10:
                    current_n = new_n
                    current_adj = trial_adj
                    if current_n > best_n:
                        best_n = current_n
                        best_adj = [row[:] for row in current_adj]
                    found = True
                    break
            
            if not found:
                break
    
    # Build edge list from best_adj
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
# EVOLVE-BLOCK-END

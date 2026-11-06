import random
import time
import sys

# EVOLVE-BLOCK-START

def is_triangle_free(adj_matrix, u, v):
    """Check if adding edge (u, v) creates a triangle."""
    # In a triangle-free graph, if (u, v) is added, there must be no common neighbor.
    # adj_matrix[i][j] is 1 if edge exists.
    common = adj_matrix[u] & adj_matrix[v]
    return common == 0

def is_3_colorable(adj_matrix, n):
    """
    Check if the graph with adjacency matrix adj_matrix (list of bitmasks) 
    is 3-colorable.
    """
    colors = [-1] * n
    # Order vertices by degree (greedy heuristic for backtracking)
    # For n=16, simple order is fine, but let's try a quick heuristic
    # Actually, for n=16, simple backtracking is extremely fast.
    
    def backtrack(idx):
        if idx == n:
            return True
        
        # Heuristic: pick uncolored vertex with most constraints? 
        # For simplicity and speed in Python, we stick to index order 
        # but we can optimize by picking the first uncolored vertex.
        # Since we add edges, the graph is built up, let's just iterate 0..n-1
        # and find the first uncolored one.
        
        # Actually, standard backtracking on fixed order is fine for n=16.
        # Let's just iterate through vertices 0 to n-1.
        # If colors[idx] is already set, we should have handled it.
        # Wait, the standard recursive backtracking assigns colors to vertices in order.
        
        # Let's implement a clean recursive backtracking that assigns color to 'idx'.
        # But 'idx' might already be colored if we jump around? No, we go in order.
        
        # Optimization: If idx is already colored, skip? No, we assign sequentially.
        
        # Check all 3 colors
        for c in range(3):
            conflict = False
            # Check neighbors
            neighbors = adj_matrix[idx]
            # Iterate through bits in neighbors
            temp = neighbors
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                if colors[j] == c:
                    conflict = True
                    break
                temp -= lsb
            
            if not conflict:
                colors[idx] = c
                if backtrack(idx + 1):
                    return True
                colors[idx] = -1
                
        return False

    return backtrack(0)

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_edges = 0
    best_adj = [0] * n
    best_edge_list = []
    
    # Initial seed: Grötzsch graph on 0..10, 11..15 isolated
    # Vertices 0-4: cycle
    # Vertices 5-9: twins of 0-4
    # Vertex 10: connected to 5-9
    groetzsch_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (0, 5), (1, 5), (1, 6), (2, 6), (2, 7), (3, 7), (3, 8), (4, 8), (4, 9),
        (5, 10), (6, 10), (7, 10), (8, 10), (9, 10)
    ]
    
    def make_adj(edge_list):
        adj = [0] * n
        for u, v in edge_list:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        return adj
    
    # Start with Grötzsch
    current_edges = list(groetzsch_edges)
    current_adj = make_adj(current_edges)
    current_count = len(current_edges)
    
    # Verify 3-colorability of start (should be true)
    if not is_3_colorable(current_adj, n):
        # Fallback to empty
        current_edges = []
        current_adj = [0] * n
        current_count = 0

    if current_count > best_edges:
        best_edges = current_count
        best_adj = current_adj[:]
        best_edge_list = current_edges[:]
        
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Random restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # Reset to a random valid triangle-free graph or Grötzsch?
        # Let's try starting from a random sparse triangle-free graph to explore different regions
        # Or just restart the greedy process from Grötzsch with different random order.
        
        # Strategy: Start with Grötzsch, then try to add edges in random order.
        # If stuck, restart.
        
        curr_edges = list(groetzsch_edges)
        curr_adj = make_adj(curr_edges)
        curr_count = len(curr_edges)
        
        # Shuffle the remaining possible edges
        remaining = [e for e in all_edges if e not in set(curr_edges)]
        random.shuffle(remaining)
        
        # Try to add each edge
        for u, v in remaining:
            if time.time() > deadline:
                break
            
            # Check triangle constraint
            if is_triangle_free(curr_adj, u, v):
                # Tentatively add
                curr_adj[u] |= 1 << v
                curr_adj[v] |= 1 << u
                
                # Check 3-colorability
                if is_3_colorable(curr_adj, n):
                    curr_edges.append((u, v))
                    curr_count += 1
                else:
                    # Revert
                    curr_adj[u] &= ~(1 << v)
                    curr_adj[v] &= ~(1 << u)
        
        if curr_count > best_edges:
            best_edges = curr_count
            best_adj = curr_adj[:]
            best_edge_list = curr_edges[:]
            
        # If we are getting close to the limit, do a final polish
        if time.time() > deadline - 10.0:
            break

    # Convert best_adj to edge list if needed, or just return best_edge_list
    # Ensure the list is in the correct format [u, v]
    return [list(e) for e in best_edge_list]

# EVOLVE-BLOCK-END

"""Program for fewest four-cycles in graphs with 31 vertices and 91 edges."""
import numpy as np
import random

def construct():
    """
    Construct a graph with 31 vertices and exactly 91 edges minimizing the number of C4 subgraphs.
    
    Approach:
    1. Build the 93-edge polarity graph on PG(2,5).
    2. Remove 2 edges to reach 91 edges, choosing removals that minimize C4 count.
    3. Perform local search (edge swaps) to further minimize C4 count.
    """
    n = 31
    target_edges = 91
    
    # Generate all points in PG(2,5)
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                pt = [x, y, z]
                for i in range(3):
                    if pt[i] != 0:
                        inv = pow(pt[i], 3, 5)  # inverse in GF(5)
                        pt = [(pt[j] * inv) % 5 for j in range(3)]
                        break
                if tuple(pt) not in seen:
                    seen.add(tuple(pt))
                    points.append(pt)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Build adjacency matrix for polarity graph
    # p . q = p[0]*q[2] + p[1]*q[0] + p[2]*q[1] == 0 mod 5
    adj = np.zeros((n, n), dtype=np.int8)
    
    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            q = points[j]
            val = (p[0] * q[2] + p[1] * q[0] + p[2] * q[1]) % 5
            if val == 0:
                adj[i, j] = 1
                adj[j, i] = 1
    
    # Count edges
    edge_count = np.sum(adj) // 2
    # Should be 93 for q=5
    
    # Helper to compute common neighbors matrix and S
    def compute_commons_and_S(adj):
        common = np.zeros((n, n), dtype=np.int32)
        S = 0
        for i in range(n):
            for j in range(i + 1, n):
                # Count common neighbors
                # Using numpy for speed, but n is small so loop is fine too
                # adj[i] & adj[j] gives boolean array of common neighbors
                c = np.sum(adj[i] & adj[j])
                common[i, j] = c
                common[j, i] = c
                S += c * (c - 1) // 2
        return common, S

    common, S = compute_commons_and_S(adj)
    
    # Function to calculate delta S for removing edge (u, v)
    # Returns delta_S, and the new common matrix (or we can just calc delta)
    # We will implement a function that calculates the change in S for a hypothetical swap/removal
    # without modifying the global state, or we modify and revert.
    # Given the small n, modifying and reverting is fine if we do it carefully.
    # But calculating delta analytically is safer.
    
    def get_delta_S_remove(u, v, common, adj):
        # Removing (u, v)
        # C_uw decreases by 1 if w is neighbor of v (w != u, v)
        # C_vw decreases by 1 if w is neighbor of u (w != u, v)
        # C_uv unchanged
        
        delta = 0
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v, w] == 1:
                # C_uw decreases by 1
                # S term for (u, w) changes from binom(c, 2) to binom(c-1, 2)
                # Delta = - (c-1)
                c = common[u, w]
                delta -= (c - 1)
            if adj[u, w] == 1:
                # C_vw decreases by 1
                c = common[v, w]
                delta -= (c - 1)
        return delta

    def get_delta_S_add(u, v, common, adj):
        # Adding (u, v)
        # C_uw increases by 1 if w is neighbor of v
        # C_vw increases by 1 if w is neighbor of u
        
        delta = 0
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v, w] == 1:
                # C_uw increases by 1
                # S term changes from binom(c, 2) to binom(c+1, 2)
                # Delta = c
                c = common[u, w]
                delta += c
            if adj[u, w] == 1:
                # C_vw increases by 1
                c = common[v, w]
                delta += c
        return delta

    # Step 1: Remove 2 edges to get to 91
    # We want to minimize S. So we want the most negative delta_S_remove.
    for _ in range(2):
        best_delta = float('inf')
        best_edge = None
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1:
                    d = get_delta_S_remove(i, j, common, adj)
                    if d < best_delta:
                        best_delta = d
                        best_edge = (i, j)
        
        if best_edge is not None:
            u, v = best_edge
            # Apply removal
            adj[u, v] = 0
            adj[v, u] = 0
            # Update common and S
            # We can just recompute, or update incrementally.
            # Recomputing is O(n^2) ~ 1000 ops. Negligible.
            common, S = compute_commons_and_S(adj)
    
    # Step 2: Local search
    # Swap (u, v) [edge] with (w, x) [non-edge]
    # Delta S = delta_remove(u, v) + delta_add(w, x)
    
    no_improvement = 0
    max_iterations = 50000
    iteration = 0
    
    # Precompute list of edges and non-edges for faster iteration?
    # Or just loop.
    
    while no_improvement < 5000 and iteration < max_iterations:
        iteration += 1
        best_total_delta = 0
        best_swap = None
        
        # Get current edges and non-edges
        edges = []
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1:
                    edges.append((i, j))
                else:
                    non_edges.append((i, j))
        
        for u, v in edges:
            for w, x in non_edges:
                # Check if it's a valid swap (not same edge)
                if (u == w and v == x) or (u == x and v == w):
                    continue
                
                d_rem = get_delta_S_remove(u, v, common, adj)
                d_add = get_delta_S_add(w, x, common, adj)
                total_delta = d_rem + d_add
                
                if total_delta < best_total_delta:
                    best_total_delta = total_delta
                    best_swap = ((u, v), (w, x))
        
        if best_swap is not None and best_total_delta < 0:
            (u, v), (w, x) = best_swap
            # Apply swap
            adj[u, v] = 0
            adj[v, u] = 0
            adj[w, x] = 1
            adj[x, w] = 1
            
            # Update common and S
            common, S = compute_commons_and_S(adj)
            no_improvement = 0
        else:
            no_improvement += 1
            
    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                result.append([i, j])
    
    assert len(result) == 91, f"Expected 91 edges, got {len(result)}"
    
    return result

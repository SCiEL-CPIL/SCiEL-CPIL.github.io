import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for overhead
    
    best_n = 39
    best_edges = []
    
    # We aim for n=47 first, then 46, then 45...
    # For n=47, min degree must be >= 8.
    # For n=46, min degree must be >= 7.
    # For n=45, min degree must be >= 6.
    
    for n_target in [47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline:
            break
            
        min_deg_req = n_target - 39
        
        # Try to find a graph for this n
        found = False
        attempts = 0
        max_attempts = 200 # Adjust based on time budget
        
        while not found and attempts < max_attempts:
            attempts += 1
            if time.time() > deadline:
                break
                
            # Initialize adjacency matrix
            # Start with a random graph with appropriate density
            # For n=47, delta>=8. Average degree ~10-12.
            # p ~ 10/47 ~ 0.2
            p = min_deg_req / n_target * 1.5 
            if p < 0.1: p = 0.1
            if p > 0.5: p = 0.5
            
            adj = np.zeros((n_target, n_target), dtype=np.uint8)
            # Generate random edges
            # Use vectorized generation for speed
            # Create all possible pairs
            # But for sparse graphs, iterating is fine, or use random sampling
            # Let's use a simple loop for initialization as n is small
            for i in range(n_target):
                for j in range(i+1, n_target):
                    if random.random() < p:
                        adj[i,j] = 1
                        adj[j,i] = 1
            
            # Repair phase
            # 1. Remove C4s
            # 2. Ensure min degree
            
            # Function to check if adding edge (u,v) creates a C4
            # A C4 is formed if there is a common neighbor w such that
            # there is another common neighbor x? No.
            # Adding (u,v) creates a C4 if there exists w such that
            # (u,w) and (v,w) are edges, AND there exists x such that
            # (u,x) and (v,x) are edges? No, that would be a C4 u-w-v-x-u.
            # Wait, a C4 is a cycle of length 4.
            # If we add edge (u,v), it creates a C4 if there is a path of length 3
            # between u and v.
            # Path u-w-x-v.
            # So we need to check if there are at least 2 common neighbors? No.
            # If u and v have common neighbors w1, w2, w3...
            # Then u-w1-v-w2-u is a C4 if w1!=w2 and edges (w1,v) and (w2,u) exist (they do by definition of common neighbor).
            # Actually, if u and v share a common neighbor w, that's a triangle (K3), not C4.
            # If u and v share TWO common neighbors w1 and w2, then u-w1-v-w2-u is a C4.
            # So, adding edge (u,v) is forbidden if |N(u) \cap N(v)| >= 2.
            
            # Function to check if removing edge (u,v) is necessary
            # An edge (u,v) is part of a C4 if there exist w1, w2 such that
            # u-w1-v-w2-u is a cycle.
            # This means w1 and w2 are common neighbors of u and v?
            # No.
            # Cycle: u-w1-v-w2-u.
            # Edges: (u,w1), (w1,v), (v,w2), (w2,u).
            # So w1 is neighbor of u and v. w2 is neighbor of u and v.
            # So u and v must have at least 2 common neighbors.
            # If u and v have >= 2 common neighbors, any edge between them is part of a C4?
            # Yes.
            
            # So the condition for C4-free is:
            # For every pair of vertices (u,v) that are NOT connected, they must have at most 1 common neighbor.
            # For every pair of vertices (u,v) that ARE connected, they must have at most 1 common neighbor?
            # No, if they are connected and have 2 common neighbors, it's a C4.
            # So for ANY pair u,v, |N(u) \cap N(v)| <= 1.
            
            # This is the key constraint: No two vertices can have more than 1 common neighbor.
            
            # Let's verify this.
            # If |N(u) \cap N(v)| >= 2, let w1, w2 be two common neighbors.
            # Then u-w1-v-w2-u is a cycle of length 4.
            # Is it simple? u != w1, w1 != v, v != w2, w2 != u.
            # Also u != v (simple graph).
            # w1 != w2 (distinct common neighbors).
            # What if u-w1-v-w2-u has repeated vertices?
            # Only if u=w2 or w1=v etc.
            # Since w1, w2 are neighbors of u and v, and graph is simple, w1 != u, w1 != v, w2 != u, w2 != v.
            # So the cycle is simple.
            
            # Therefore, the graph is C4-free if and only if for all u != v, |N(u) \cap N(v)| <= 1.
            
            # This simplifies the check significantly.
            
            # We need to construct a graph such that:
            # 1. For all u != v, |N(u) \cap N(v)| <= 1.
            # 2. min degree >= n - 39.
            
            # Let's repair the graph.
            # We maintain the adjacency matrix.
            # We iterate through pairs or vertices to fix violations.
            
            # A violation is a pair (u,v) with |N(u) \cap N(v)| >= 2.
            # To fix, we can remove an edge incident to u or v that is part of the intersection.
            # Actually, if |N(u) \cap N(v)| = k >= 2, we must remove edges until k <= 1.
            # Which edges to remove?
            # Removing edge (u, w) where w is a common neighbor reduces the intersection size for pairs (u, v) and (w, v)??
            # No.
            # If we remove edge (u, w), then w is no longer in N(u).
            # So w is no longer in N(u) \cap N(v).
            # This reduces the common neighbor count for pair (u,v) by 1.
            # It also affects other pairs.
            
            # Strategy:
            # While there is a violation (pair u,v with >=2 common neighbors):
            #   Pick one such pair.
            #   Pick a common neighbor w.
            #   Remove edge (u,w) or (v,w).
            #   To preserve degree, we might prefer removing the edge that is "less critical" for degree constraints.
            #   Or just remove randomly.
            
            # After removing C4s, we might have low degree.
            # Then we add edges to increase degree, ensuring no new C4s are formed.
            # Adding edge (u,v) is safe if |N(u) \cap N(v)| <= 1.
            
            # Let's implement this repair loop.
            
            valid = True
            # Initial check for C4s
            # We can check all pairs. n=47, pairs ~ 1100.
            # Intersection check is O(n) or vectorized.
            
            def has_c4(adj):
                # Check all pairs
                for i in range(n_target):
                    for j in range(i+1, n_target):
                        # Count common neighbors
                        # adj[i] & adj[j]
                        # np.count_nonzero(adj[i] & adj[j])
                        # This is slow in python loop.
                        # Vectorize:
                        pass
                # Better: Compute all pairwise intersections?
                # Common neighbors matrix C where C[i,j] = (adj[i] & adj[j]).sum()
                # C = adj @ adj.T
                # If any C[i,j] >= 2 for i!=j, then C4 exists.
                # Note: C[i,i] is degree.
                # We need to check off-diagonal.
                return False # Placeholder
            
            # Let's use the matrix multiplication trick.
            # C = adj @ adj.T
            # For i != j, C[i,j] is the number of common neighbors.
            # If max(C[i,j]) >= 2 for any i!=j, C4 exists.
            
            def check_and_fix(adj):
                # Returns True if fixed and valid, False if stuck or timeout
                while True:
                    if time.time() > deadline:
                        return False
                    
                    # Check for C4
                    common = adj @ adj.T
                    # Mask diagonal
                    np.fill_diagonal(common, 0)
                    
                    # Find max common neighbors
                    max_common = np.max(common)
                    if max_common <= 1:
                        # No C4s
                        break
                    
                    # Find a violating pair
                    # Get indices of max
                    # We just need one
                    idx = np.unravel_index(np.argmax(common), common.shape)
                    u, v = idx[0], idx[1]
                    if u == v: continue # Should not happen due to mask
                    
                    # We need to reduce common neighbors of u and v to <= 1
                    # Current count = max_common
                    # We need to remove (max_common - 1) edges
                    # Common neighbors are indices k where adj[u,k]==1 and adj[v,k]==1
                    common_neighbors = np.nonzero(adj[u] & adj[v])[0]
                    
                    # Remove edges to reduce count
                    # We want to keep degrees high.
                    # Maybe remove edge from the vertex with higher degree?
                    # Or random.
                    # Let's remove edges to random common neighbors until count <= 1
                    while len(common_neighbors) > 1:
                        if time.time() > deadline:
                            return False
                        # Pick a common neighbor to disconnect from
                        # Choose u or v
                        if random.random() < 0.5:
                            # Remove (u, w)
                            w = common_neighbors[0] # Just take first, or random
                            adj[u, w] = 0
                            adj[w, u] = 0
                        else:
                            # Remove (v, w)
                            w = common_neighbors[0]
                            adj[v, w] = 0
                            adj[w, v] = 0
                        
                        # Update common neighbors list
                        common_neighbors = np.nonzero(adj[u] & adj[v])[0]
                    
                    # Loop to re-check, as removing edges might create other issues?
                    # No, removing edges only decreases common neighbors.
                    # But it might create low degree.
            
            # Execute repair
            if check_and_fix(adj):
                # Now ensure min degree
                degrees = np.sum(adj, axis=1)
                while np.min(degrees) < min_deg_req:
                    if time.time() > deadline:
                        break
                    # Find vertex with min degree
                    v = np.argmin(degrees)
                    # Find non-neighbors
                    non_neighbors = np.nonzero(adj[v] == 0)[0]
                    # Exclude self
                    non_neighbors = non_neighbors[non_neighbors != v]
                    
                    if len(non_neighbors) == 0:
                        # Complete graph, but C4 free? No.
                        break
                    
                    # Try to add edge to a non-neighbor
                    # Safe if |N(v) \cap N(u)| <= 1
                    added = False
                    # Shuffle non-neighbors to try random ones
                    random.shuffle(non_neighbors)
                    for u in non_neighbors:
                        # Check if adding (v,u) creates C4
                        # Common neighbors of v and u
                        common_count = np.sum(adj[v] & adj[u])
                        if common_count <= 1:
                            # Safe to add
                            adj[v, u] = 1
                            adj[u, v] = 1
                            degrees[v] += 1
                            degrees[u] += 1
                            added = True
                            break
                    
                    if not added:
                        # Cannot add any edge without creating C4
                        # This means we are stuck.
                        break
                
                # Final check
                if np.min(degrees) >= min_deg_req:
                    # Verify no C4
                    common = adj @ adj.T
                    np.fill_diagonal(common, 0)
                    if np.max(common) <= 1:
                        # Valid graph found
                        found = True
                        best_n = n_target
                        # Extract edges
                        edges = []
                        for i in range(n_target):
                            for j in range(i+1, n_target):
                                if adj[i,j] == 1:
                                    edges.append([i, j])
                        best_edges = edges
                        break # Move to next n_target? No, we want max n.
                        # Since we iterate 47, 46..., if we find 47, we can stop.
                        # But maybe 47 is hard. If we find 46, we should stop?
                        # The task says "Maximize n".
                        # So if we find 47, we are done.
                        # If we find 46, we should try to see if 47 is possible?
                        # We are iterating from 47 down.
                        # So if we find 47, we break out of the outer loop.
                        # If we don't find 47 in time, we try 46.
                        # If we find 46, we should stop?
                        # Yes, because we want the MAX n.
                        # If we found 46, we know 47 failed (or timed out).
                        # But maybe 47 was possible and we just didn't find it.
                        # Given the time limit, finding 46 is a good result.
                        # We can break the outer loop once we find a valid graph.
                        # Because we start from the highest n.
                        break
            
            # End while attempts
        
        # If found a graph for this n_target, we can stop searching for lower n
        if found:
            break
        
    # Return the best result
    return {'n': best_n, 'edges': best_edges}

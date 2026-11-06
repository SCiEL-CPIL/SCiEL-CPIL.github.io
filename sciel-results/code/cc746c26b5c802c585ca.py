import numpy as np
import random
from itertools import combinations

def construct():
    n = 20
    
    # 1. Spectral Heuristic Initialization
    # Generate a random symmetric matrix
    M = np.random.randn(n, n)
    M = (M + M.T) / 2.0
    
    # Compute the top eigenvector
    # eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    v_top = eigenvectors[:, -1]
    
    # Construct adjacency matrix based on the outer product of the top eigenvector
    # X_ij = v_i * v_j
    X = np.outer(v_top, v_top)
    
    # Add small noise to break symmetry and avoid pure cliques/bipartite structures
    noise = np.random.randn(n, n) * 0.1
    X_noisy = X + noise
    # Ensure symmetry
    X_noisy = (X_noisy + X_noisy.T) / 2.0
    np.fill_diagonal(X_noisy, 0)
    
    # Threshold at the mean to create a dense random-like starting point
    threshold = np.mean(X_noisy[X_noisy > 0]) if np.any(X_noisy > 0) else 0
    # If mean is 0 or negative, use a small positive value to ensure some edges
    if threshold <= 0:
        threshold = 0.01
        
    adj = (X_noisy > threshold).astype(np.uint8)
    # Ensure no self-loops
    np.fill_diagonal(adj, 0)
    
    # 2. Repair Loop with Optimized Common Neighbor Counting
    # We use bitmasks for fast common neighbor counting
    # Represent adjacency as integers for bit operations
    adj_int = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i, j]:
                mask |= (1 << j)
        adj_int[i] = mask
        
    def get_common_neighbors(u, v, w):
        # Count common neighbors of u, v, w
        # Common neighbors are bits set in adj[u], adj[v], and adj[w]
        # Exclude u, v, w themselves? 
        # In a simple graph, adj[u] does not have bit u set.
        # But if u is a neighbor of v, bit u is in adj[v].
        # We need neighbors c such that c != u, c != v, c != w.
        common_mask = adj_int[u] & adj_int[v] & adj_int[w]
        # Clear bits for u, v, w
        common_mask &= ~(1 << u)
        common_mask &= ~(1 << v)
        common_mask &= ~(1 << w)
        return common_mask

    max_iterations = 200
    for _ in range(max_iterations):
        violation_found = False
        # Find a violating triple
        # To speed up, we can iterate over triples.
        # n=20, triples = 1140.
        # For each triple, count common neighbors.
        # If count >= 3, violation.
        
        for u in range(n):
            for v in range(u + 1, n):
                for w in range(v + 1, n):
                    common_mask = get_common_neighbors(u, v, w)
                    # Count bits
                    count = bin(common_mask).count('1')
                    if count >= 3:
                        violation_found = True
                        # Get the indices of the common neighbors
                        common_neighbors = []
                        for c in range(n):
                            if common_mask & (1 << c):
                                common_neighbors.append(c)
                        
                        # Pick 3 common neighbors
                        c1, c2, c3 = common_neighbors[0], common_neighbors[1], common_neighbors[2]
                        
                        # Remove a random edge from the K3,3 subgraph
                        # The K3,3 is between {u, v, w} and {c1, c2, c3}
                        # Edges are (u,c1), (u,c2), (u,c3), (v,c1)... etc.
                        # We remove one edge to break this specific K3,3.
                        # Removing (u, c1) is a good choice.
                        adj[u, c1] = 0
                        adj[c1, u] = 0
                        adj_int[u] &= ~(1 << c1)
                        adj_int[c1] &= ~(1 << u)
                        break
                if violation_found:
                    break
            if violation_found:
                break
        
        if not violation_found:
            break

    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append([i, j])
                
    return edges

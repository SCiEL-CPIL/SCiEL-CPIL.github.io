"""Program using spectral gap heuristic to guide edge insertion for C6-free graphs."""
import numpy as np
import time
import sys

def _has_c6(adj, u, v):
    """
    Check if adding edge (u,v) creates a C6.
    We need to check if there is a simple path of length 5 from u to v in the current graph.
    """
    n = len(adj)
    # adj is a numpy array of int8/bool
    # Convert to list of lists of neighbors for faster iteration in Python DFS
    # Or stick to numpy for matrix ops?
    # For DFS, Python lists are often faster for small n if we avoid numpy overhead in the loop.
    # But we can use a bitset approach with integers.
    
    # Precompute neighbors as bitmasks?
    # n=40, fits in 64-bit int.
    
    # Let's use a standard DFS with a visited bitmask (int).
    
    def dfs(node, depth, visited_mask):
        if depth == 5:
            return node == v
        
        if depth == 4:
            # Next step must be v
            # Check if v is neighbor of node and not visited
            if (adj[node][v] == 1) and not (visited_mask & (1 << v)):
                return True
            return False
        
        # Iterate neighbors
        # To speed up, we can iterate over set bits in adj[node]
        # But adj is numpy. 
        # Let's convert adj to a list of lists of neighbors once outside?
        # No, adj changes. 
        # We can pass neighbors list or use adj directly.
        
        # Optimization: 
        # If we have a global neighbors list, we can use it.
        # But in this function, we only have adj.
        
        # Let's just loop 0..n-1. n=40 is small.
        for nxt in range(n):
            if nxt == u:
                continue
            if nxt == v and depth < 4:
                continue
            
            if adj[node][nxt]:
                if not (visited_mask & (1 << nxt)):
                    if dfs(nxt, depth + 1, visited_mask | (1 << nxt)):
                        return True
        
        return False

    visited_mask = 1 << u
    return dfs(u, 0, visited_mask)

def construct():
    n = 40
    start_time = time.time()
    timeout = 110
    
    # Start with 8 disjoint copies of K5
    adj = np.zeros((n, n), dtype=np.int8)
    edges = []
    
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj[u][v] = 1
                adj[v][u] = 1
                edges.append([u, v])
                
    # Precompute all possible edges
    all_possible = set()
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] == 0:
                all_possible.add((i, j))
                
    # Main loop
    while all_possible:
        if time.time() - start_time > timeout:
            break
        
        # Compute spectral properties
        # Use eigh for symmetric matrix
        try:
            # Only need top 2 eigenvalues/vectors
            # numpy.linalg.eigh returns all. 
            # We can use scipy.sparse.linalg.eigsh for speed if available, but standard lib is preferred.
            # For 40x40, eigh is fast enough.
            vals, vecs = np.linalg.eigh(adj.astype(float))
            
            # eigh returns in ascending order. 
            # Largest is last, second largest is second last.
            v1 = vecs[:, -1]
            v2 = vecs[:, -2]
        except Exception:
            break
            
        # Score all candidates
        # Score = v1[u]*v1[v] - v2[u]*v2[v]
        # We want to maximize this score.
        
        candidates = []
        for u, v in all_possible:
            score = v1[u] * v1[v] - v2[u] * v2[v]
            candidates.append((score, u, v))
            
        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Try to add the best valid edge
        added = False
        for score, u, v in candidates:
            if time.time() - start_time > timeout:
                break
                
            if _has_c6(adj, u, v):
                continue
                
            # Add edge
            adj[u][v] = 1
            adj[v][u] = 1
            edges.append([u, v])
            all_possible.remove((u, v))
            added = True
            break # Restart loop to update eigenvectors
            
        if not added:
            # No valid edges found in this round?
            # If candidates was empty, we broke out.
            # If all candidates failed C6 check, we should break.
            # But maybe some lower scored edges are valid?
            # The loop above tries ALL candidates in sorted order until one is added.
            # If none is added, it means NO edge in all_possible is valid.
            break
            
    return edges

import numpy as np
from collections import deque

def sign_edges(graph, budget):
        n = graph['n']
        edges = graph['edges']
        m = n // 2
        
        # Build adjacency list with edge indices
        adj = {i: [] for i in range(n)}
        for idx, (u, v) in enumerate(edges):
            adj[u].append((v, idx))
            adj[v].append((u, idx))
            
        # Find spanning tree
        visited = [False] * n
        tree_edges = set()
        queue = deque([0])
        visited[0] = True
        
        while queue:
            u = queue.popleft()
            for v, idx in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    tree_edges.add(idx)
                    queue.append(v)
                    
        # Identify non-tree edges
        non_tree_edges = [idx for idx in range(len(edges)) if idx not in tree_edges]
        num_non_tree = len(non_tree_edges)
        
        # The number of non-tree edges should be m+1
        # Iterate over all 2^(m+1) patterns
        best_radius = float('inf')
        best_signs = None
        
        # We can skip the all +1 case for non-tree edges because it's balanced (disconnected)
        # But wait, is it possible that the all +1 case is the only one?
        # No, we need connected lift.
        
        for mask in range(1, 1 << num_non_tree): # Start from 1 to skip all 0s (all +1)
            # Create signs array
            signs = np.ones(len(edges), dtype=int)
            
            # Assign -1 to non-tree edges based on mask
            for i, edge_idx in enumerate(non_tree_edges):
                if (mask >> i) & 1:
                    signs[edge_idx] = -1
            
            # Construct Signed Adjacency Matrix
            A = np.zeros((n, n))
            for idx, (u, v) in enumerate(edges):
                s = signs[idx]
                A[u, v] = s
                A[v, u] = s # Undirected, symmetric
            
            # Compute spectral radius
            # Since A is symmetric, eigvalsh is fast
            eigvals = np.linalg.eigvalsh(A)
            radius = np.max(np.abs(eigvals))
            
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
                
        # If no valid signs found (should not happen for m>=4), fallback
        if best_signs is None:
             # Try random or all -1?
             # For m=4, there are 2^5 - 1 = 31 valid patterns.
             # So this shouldn't happen.
             best_signs = np.ones(len(edges), dtype=int)
             
        return best_signs

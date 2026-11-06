"""Program using backtracking to find the minimum threshold K for a Hamiltonian cycle."""
import random
import sys

def construct():
    n = 24
    # Generate the pinned edge list with random.Random(614)
    rng = random.Random(614)
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((i, j))
    rng.shuffle(all_edges)
    
    # Build edge index lookup: edge -> position (1-based)
    edge_pos = {}
    for idx, (u, v) in enumerate(all_edges, 1):
        edge_pos[(u, v)] = idx
        edge_pos[(v, u)] = idx

    def find_hamiltonian_cycle(K):
        """
        Find a Hamiltonian cycle in the graph formed by edges with position <= K.
        Returns a list of vertices representing the cycle (starting with 0) or None.
        """
        # Build adjacency list for edges with pos <= K
        adj = {v: [] for v in range(n)}
        for (u, v) in all_edges:
            pos = edge_pos[(u, v)]
            if pos <= K:
                adj[u].append(v)
                adj[v].append(u)
        
        # Sort neighbors by degree (Warnsdorff's rule heuristic) to prune faster
        # We do this dynamically inside the DFS? No, static sorting based on degree in G_K is a good heuristic.
        for v in range(n):
            # Sort by degree of the neighbor in G_K
            adj[v].sort(key=lambda x: len(adj[x]))
            
        # If any vertex has degree < 2, no Hamiltonian cycle exists
        for v in range(n):
            if len(adj[v]) < 2:
                return None

        path = [0]
        visited = {0}
        
        # Increase recursion limit just in case, though n=24 is small
        sys.setrecursionlimit(10000)
        
        def dfs(current):
            if len(path) == n:
                # Check if we can close the loop
                if 0 in adj[current]:
                    return True
                return False
            
            for neighbor in adj[current]:
                if neighbor not in visited:
                    # Additional pruning: if neighbor is 0 and we haven't visited all nodes, skip
                    if neighbor == 0 and len(path) < n:
                        continue
                    
                    # Pruning: If the unvisited neighbors of 'neighbor' (excluding current) are not enough to complete the path?
                    # Simple check: if neighbor is the last node to be added, it must connect to 0.
                    
                    path.append(neighbor)
                    visited.add(neighbor)
                    
                    if dfs(neighbor):
                        return True
                    
                    path.pop()
                    visited.remove(neighbor)
            
            return False
        
        if dfs(0):
            return path
        return None

    # Binary search for minimum K
    lo = n  # at least n edges needed
    hi = n * (n - 1) // 2  # all edges
    
    # First check if it's feasible with all edges
    if find_hamiltonian_cycle(hi) is None:
        # Should not happen, but fallback
        return list(range(n))
    
    while lo < hi:
        mid = (lo + hi) // 2
        if find_hamiltonian_cycle(mid) is not None:
            hi = mid
        else:
            lo = mid + 1
            
    K = lo
    
    # Now find the actual Hamiltonian cycle using K
    cycle = find_hamiltonian_cycle(K)
    
    if cycle is None:
        # Fallback
        return list(range(n))
        
    return cycle

import random
import time

def construct():
    # Target: Maximize n <= 41
    # Constraints: No triangles, no independent set of size 10.
    
    def is_valid(n, edges):
        # Check for triangles
        # Build adjacency matrix for speed
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # Triangle check
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    for k in range(j + 1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        
        # Independent set check (size 10)
        # Use branch and bound or simple heuristic?
        # For n=39, checking independent set of size 10 is hard.
        # We can use a greedy upper bound or exact search with pruning.
        
        def has_independent_set(size):
            # Branch and bound
            # vertices 0..n-1
            # remaining vertices set
            # current independent set size
            
            # Simplified: Use a max independent set algorithm
            # If max independent set < 10, return False (valid)
            # If max independent set >= 10, return True (invalid)
            
            # We want to know if there is an independent set of size 10.
            # This is equivalent to checking if the clique number of the complement graph is >= 10.
            # Complement graph edges are non-edges of G.
            
            # Let's use a simple DFS with pruning.
            # Order vertices by degree in complement (or G)?
            
            # For speed, we can use a greedy approximation first.
            # But we need exactness for the final result.
            
            # Given time limit, we might need a fast heuristic for intermediate steps,
            # but the final returned graph MUST be valid.
            
            # Let's implement a fast exact check for small n or use a library if available?
            # No library.
            
            # We'll use a recursive search with pruning.
            # To speed it up, we can sort vertices by degree in the complement graph.
            
            # Build complement adjacency
            comp_adj = [[False] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if not adj[i][j]:
                        comp_adj[i][j] = True
                        comp_adj[j][i] = True
            
            # Max clique in comp_adj
            # Bron-Kerbosch with pivot
            
            best_clique_size = 0
            
            def bron_kerbosch(R, P, X):
                nonlocal best_clique_size
                if not P and not X:
                    if len(R) > best_clique_size:
                        best_clique_size = len(R)
                    return
                
                # Pivot selection
                if P:
                    # Choose pivot u in P U X with max degree in P
                    u = max(P | X, key=lambda x: sum(comp_adj[x][y] for y in P))
                    candidates = [v for v in P if not comp_adj[v][u]]
                else:
                    candidates = P
                
                for v in candidates:
                    bron_kerbosch(R | {v}, P & {w for w in P if comp_adj[v][w]}, X & {w for w in X if comp_adj[v][w]})
                    P.remove(v)
                    X.add(v)
                    
                    # Pruning: if max possible size is less than 10, stop
                    if best_clique_size >= 10:
                        return
                    # Simple bound: len(R) + len(P) < 10
                    if len(R) + len(P) < 10:
                        return

            bron_kerbosch(set(), set(range(n)), set())
            return best_clique_size >= 10

        return not has_independent_set(10)

    # Search strategy
    # Try n from 39 down to 27
    # Use local search to find a valid graph
    
    for n in range(39, 26, -1):
        # Initialize with a random graph or structured graph
        # Structured: Circulant with random S
        found = False
        for attempt in range(50):
            # Generate random graph
            # Start with sparse, add edges?
            # Or start with dense, remove triangles?
            
            # Let's start with a random graph and remove triangles
            edges = []
            adj = [[False] * n for _ in range(n)]
            
            # Add random edges
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        edges.append([i, j])
                        adj[i][j] = True
                        adj[j][i] = True
            
            # Remove triangles
            changed = True
            while changed:
                changed = False
                new_edges = []
                for u, v in edges:
                    # Check if adding this edge creates a triangle
                    # Actually, we just check if current graph has triangles
                    pass
                
                # Simpler: Iterate and remove one edge from each triangle
                to_remove = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i][j]:
                            for k in range(j + 1, n):
                                if adj[i][k] and adj[j][k]:
                                    to_remove.add((i, j))
                                    to_remove.add((i, k))
                                    to_remove.add((j, k))
                                    break
                            if (i, j) in to_remove:
                                break
                    if (i, j) in to_remove:
                        break
                
                if to_remove:
                    for u, v in to_remove:
                        if u < v:
                            adj[u][v] = False
                            adj[v][u] = False
                    edges = [[u, v] for u, v in edges if not (adj[u][v] == False and u < v)]
                    # Rebuild edges list properly
                    edges = [[u, v] for u in range(n) for v in range(u + 1, n) if adj[u][v]]
                    changed = True
            
            # Check if valid
            if is_valid(n, edges):
                return {'n': n, 'edges': edges}
        
        # If not found for this n, try smaller n
        # Note: is_valid is expensive. We might need to optimize it.
        
    # Fallback: Return the Andrasfai graph on 26 vertices
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}

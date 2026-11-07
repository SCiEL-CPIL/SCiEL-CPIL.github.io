import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets for the original graph
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Set of uncovered edges, represented as frozensets for hashability
    uncovered = set()
    for u, v in edges:
        uncovered.add(frozenset((u, v)))
        
    bicliques = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick an edge to start the biclique
        # Heuristic: pick an edge with high degree in the uncovered graph
        # For simplicity, we can just pick the first one, but let's try to pick a good one
        # We'll pick an edge (u, v) from uncovered
        # To improve, we can try a few random edges and pick the one that expands best
        # But for now, let's just pick the first one for speed and determinism
        # Actually, let's pick the edge with the maximum sum of degrees in the uncovered graph
        # This might be too slow if we do it every time. Let's just pick the first one.
        
        seed_edge = next(iter(uncovered))
        u, v = tuple(seed_edge)
        
        # Initialize biclique
        L = {u}
        R = {v}
        
        # Expand the biclique
        changed = True
        while changed:
            changed = False
            # Try to add to L
            new_L = set(L)
            for w in range(n):
                if w not in L and w not in R:
                    # Check if w is connected to all in R in the uncovered graph
                    # Note: w must be connected to all r in R in the ORIGINAL graph
                    # AND the edges (w, r) must be in the uncovered set
                    # Wait, the biclique must be a subset of the original graph.
                    # But we are partitioning the uncovered edges.
                    # So the biclique edges must be in the uncovered set.
                    # So for w to be in L, it must be connected to all r in R in the uncovered set.
                    can_add = True
                    for r in R:
                        if frozenset((w, r)) not in uncovered:
                            can_add = False
                            break
                    if can_add:
                        new_L.add(w)
                        changed = True
            L = new_L
            
            # Try to add to R
            new_R = set(R)
            for w in range(n):
                if w not in R and w not in L:
                    can_add = True
                    for l in L:
                        if frozenset((l, w)) not in uncovered:
                            can_add = False
                            break
                    if can_add:
                        new_R.add(w)
                        changed = True
            R = new_R
            
        # Remove all edges between L and R from uncovered
        for l in L:
            for r in R:
                edge = frozenset((l, r))
                if edge in uncovered:
                    uncovered.remove(edge)
                    
        # Add the biclique to the solution
        # Ensure non-empty disjoint sides
        if L and R:
            bicliques.append([list(L), list(R)])
            
    # If we timed out, we might have uncovered edges left.
    # We must cover them to be valid.
    # We can cover them one by one with single-edge bicliques.
    if uncovered:
        for edge in uncovered:
            u, v = tuple(edge)
            bicliques.append([[u], [v]])
            
    return bicliques

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix for fast lookup
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True

    # Bron-Kerbosch with pivot to find maximum clique
    best_clique = []
    
    # Greedy coloring for upper bound
    def greedy_coloring(vertices):
        # Returns a list of colors and the max color used
        colors = {}
        max_color = 0
        # Simple greedy: assign lowest available color
        # For upper bound in BK, we usually order by color
        # This is a simplified version for pruning
        pass # We will use a standard BK implementation without complex coloring first, as N=24 is small
        
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        
        # Choose pivot u from P union X
        # u with max degree in P
        u = -1
        max_deg = -1
        for v in P:
            deg = 0
            for w in P:
                if adj[v][w]:
                    deg += 1
            if deg > max_deg:
                max_deg = deg
                u = v
        if u == -1:
            for v in X:
                deg = 0
                for w in P:
                    if adj[v][w]:
                        deg += 1
                if deg > max_deg:
                    max_deg = deg
                    u = v
        if u == -1:
            u = next(iter(P)) if P else next(iter(X))

        # Recurse on vertices in P that are not neighbors of u
        # If u is in P, neighbors are those in P adjacent to u
        # We iterate over P \ N(u)
        candidates = [v for v in P if not adj[v][u]]
        
        for v in candidates:
            bron_kerbosch(
                R | {v}, 
                P & {w for w in P if adj[v][w]}, 
                X & {w for w in X if adj[v][w]}
            )
            P.remove(v)
            X.add(v)

    # Initialize
    all_vertices = set(range(n))
    bron_kerbosch(set(), all_vertices, set())
    
    # If no clique found (should not happen), fallback
    if not best_clique:
        best_clique = [0, 1]
        
    # Ensure at least 2 branches
    if len(best_clique) < 2:
        # Find an edge
        for u, v in edges:
            best_clique = [u, v]
            break

    branches = sorted(best_clique)
    paths = []
    for u, v in combinations(branches, 2):
        paths.append([u, v])
        
    return {'branches': branches, 'paths': paths}

# EVOLVE-BLOCK-END

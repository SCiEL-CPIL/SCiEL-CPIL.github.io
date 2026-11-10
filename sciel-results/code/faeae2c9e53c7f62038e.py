import sys
from itertools import combinations

# Increase recursion limit just in case
sys.setrecursionlimit(10000)

def construct():
    # P is global from the seed
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix or sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Bron-Kerbosch algorithm with pivoting to find Max Clique
    max_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal max_clique
        
        # Pruning: if the size of R plus the size of P cannot exceed the current max clique size
        if len(R) + len(P) <= len(max_clique):
            return

        if not P and not X:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        
        # Choose pivot u from P union X to minimize branching
        # We want u to have the most neighbors in P, so that P \ N(u) is as small as possible
        U = P.union(X)
        if not U:
            return
            
        max_neighbors_in_P = -1
        u = None
        for v in U:
            # Count how many vertices in P are adjacent to v
            count = 0
            for w in P:
                if w in adj[v]:
                    count += 1
            if count > max_neighbors_in_P:
                max_neighbors_in_P = count
                u = v
                
        # If no pivot found (should not happen if U is not empty), fallback
        if u is None:
            u = next(iter(U))
            
        # Branch on vertices in P \ N(u)
        candidates = [v for v in P if v not in adj[u]]
        
        # Sort candidates by degree in P to find large cliques early (better pruning)
        candidates.sort(key=lambda x: len(adj[x] & P), reverse=True)
        
        for v in candidates:
            bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
            
    bron_kerbosch(set(), set(range(n)), set())
    
    # Fallback if no clique found (should not happen, at least 1)
    if not max_clique:
        max_clique = [0]
        
    # If max clique size is 1, we need at least 2 branches.
    # The problem states "at least 2 branches".
    # A clique of size 1 is not enough.
    # A clique of size 2 is an edge.
    # If max_clique has size 1, find any edge.
    if len(max_clique) < 2:
        # Find any edge
        for u, v in edges:
            max_clique = [u, v]
            break
            
    branches = list(max_clique)
    
    # Construct paths
    # For a clique, the path between any two branches is just the edge [u, v]
    paths = []
    for u, v in combinations(branches, 2):
        paths.append([u, v])
        
    return {'branches': branches, 'paths': paths}

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

# The function construct must be defined after P or use P from global scope.
# In the provided seed, P is defined before construct.
# I will keep the structure.

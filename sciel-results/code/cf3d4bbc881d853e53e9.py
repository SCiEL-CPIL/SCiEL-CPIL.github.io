import time
import itertools
import random

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # 1. Find Max Clique
    def find_max_clique():
        max_clique = []
        # Bron-Kerbosch with pivoting
        def bron_kerbosch(R, P, X):
            nonlocal max_clique
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) > len(max_clique):
                    max_clique = list(R)
                return
            # Choose pivot u from P union X with max degree
            u = max(P.union(X), key=lambda x: len(adj[x] & (P.union(X))))
            candidates = P - adj[u]
            for v in list(candidates):
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bron_kerbosch(set(), set(range(n)), set())
        return max_clique

    best_branches = find_max_clique()
    if not best_branches:
        best_branches = [0, 1]
        
    # Check if we have time to try for larger topological minors
    # A clique is a valid subdivision.
    # Can we find k > len(best_branches) branches?
    
    def is_clique(branches):
        for i in range(len(branches)):
            for j in range(i+1, len(branches)):
                if branches[j] not in adj[branches[i]]:
                    return False
        return True

    # Try to extend the clique to a topological minor
    # Heuristic: Try to add vertices one by one
    # This is complex, so we stick to finding the largest clique first.
    # If max clique is 4, score is 4/3 = 1.33.
    # If max clique is 5, score is 5/3 = 1.66.
    
    # Let's also try a random search for larger sets that might form a subdivision
    # even if not a clique.
    # Given the small N, we can try to verify sets of size K+1, K+2
    
    def find_disjoint_paths(branches):
        # Returns list of paths if found, else None
        # This is a simplified check: just return direct edges if clique
        # For non-cliques, we'd need a path finder.
        # For now, assume we only return cliques.
        pass

    # To improve further, we can try to find a K_5 topological minor
    # if max clique is 4.
    # But implementing a robust K_k topological minor finder is hard.
    # Given the constraints, returning the max clique is a solid strategy.
    
    # Ensure types are correct
    branches = list(best_branches)
    paths = []
    for i in range(len(branches)):
        for j in range(i+1, len(branches)):
            u, v = branches[i], branches[j]
            # Since it's a clique, the path is just [u, v]
            paths.append([u, v])
            
    return {'branches': branches, 'paths': paths}
# EVOLVE-BLOCK-END

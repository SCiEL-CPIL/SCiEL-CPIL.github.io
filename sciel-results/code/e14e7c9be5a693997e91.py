import time
from collections import defaultdict, deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    deadline = time.time() + 50.0 # Safety margin
    n = P['n']
    
    # Build adjacency sets
    adj_sets = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj_sets[u].add(v)
        adj_sets[v].add(u)
        
    # Find Max Clique using Bron-Kerbosch
    best_clique = set()
    
    def bron_kerbosch(R, P_set, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = set(R)
            return
        
        if P_set:
            # Pivot selection
            pivot = max(P_set, key=lambda v: len(P_set & adj_sets[v]) + len(X & adj_sets[v]))
            candidates = P_set - adj_sets[pivot]
        else:
            candidates = X
            
        for v in list(candidates):
            if time.time() > deadline:
                return
            bron_kerbosch(R | {v}, P_set & adj_sets[v], X & adj_sets[v])
            P_set.remove(v)
            X.add(v)
            
    bron_kerbosch(set(), set(range(n)), set())
    
    # If no clique found or too small, use greedy
    if len(best_clique) < 2:
        # Greedy fallback
        best_clique = set()
        for v in sorted(range(n), key=lambda x: -len(adj_sets[x])):
            if all(v in adj_sets[u] for u in best_clique):
                best_clique.add(v)
                
    if len(best_clique) < 2:
        # Last resort
        return {'branches': [0, 1], 'paths': [[0, 1]]}
        
    branches = sorted(list(best_clique))
    paths = []
    for i in range(len(branches)):
        for j in range(i+1, len(branches)):
            u = branches[i]
            v = branches[j]
            # Since they are in a clique, they are adjacent
            paths.append([u, v])
            
    return {'branches': branches, 'paths': paths}

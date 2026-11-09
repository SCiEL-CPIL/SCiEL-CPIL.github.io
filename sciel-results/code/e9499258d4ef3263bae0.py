"""Rewritten program: exact search over all subsets of size >= 8."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
_adj = [set() for _ in range(_N)]
for (u, v) in P['edges']:
    _adj[u].add(v)
    _adj[v].add(u)

def construct():
    best_score = -1
    best_vertices = list(range(8))
    
    # We search over all subsets of vertices of size >= 8.
    # This is equivalent to searching over removed sets R of size <= 9.
    for r in range(0, 10):
        for R in itertools.combinations(range(_N), r):
            R_set = frozenset(R)
            V_prime = [v for v in range(_N) if v not in R_set]
            if len(V_prime) < 8:
                continue
            
            # Calculate degrees in induced subgraph V_prime
            degs = set()
            for v in V_prime:
                d = 0
                for u in V_prime:
                    if u != v and u in _adj[v]:
                        d += 1
                degs.add(d)
            
            score = len(degs)
            if score > best_score:
                best_score = score
                best_vertices = V_prime
    
    return [int(x) for x in best_vertices]
# EVOLVE-BLOCK-END

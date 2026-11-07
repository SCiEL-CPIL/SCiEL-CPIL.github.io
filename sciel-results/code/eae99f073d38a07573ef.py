"""Improved program: find maximum clique (branch set with direct-edge paths)."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    es = set()
    adj = [set() for _ in range(n)]
    for u, v in edges:
        es.add((u, v))
        es.add((v, u))
        adj[u].add(v)
        adj[v].add(u)

    def has_edge(a, b):
        return b in adj[a]

    # Bron-Kerbosch with pivot (Tomita improvement)
    best_clique = []
    deadline = time.time() + 80.0  # safety margin before 90s kill

    def bk(R, P_set, X_set):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P_set and not X_set:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        if not P_set:
            return
        # Pivot selection: choose u in P_set ∪ X_set with max |P_set ∩ adj[u]|
        pivot_candidates = list(P_set) + list(X_set)
        best_u = max(pivot_candidates, key=lambda u: len(P_set & adj[u]))
        # Iterate over P_set \ adj[best_u]
        candidates = list(P_set - adj[best_u])
        for v in candidates:
            if time.time() > deadline:
                return
            R.append(v)
            new_P = P_set & adj[v]
            new_X = X_set & adj[v]
            bk(R, new_P, new_X)
            R.pop()
            P_set.discard(v)
            X_set.add(v)

    bk([], set(range(n)), set())

    # Fallback: ensure at least 2 branches
    if len(best_clique) < 2:
        u, v = edges[0]
        best_clique = [u, v]

    branches = sorted(int(x) for x in best_clique)
    paths = []
    for a, b in combinations(branches, 2):
        paths.append([int(a), int(b)])

    return {'branches': branches, 'paths': paths}
# EVOLVE-BLOCK-END

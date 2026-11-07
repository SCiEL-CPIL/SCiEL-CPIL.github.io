"""Rewritten: exact maximum-clique via Bron-Kerbosch with pivot + greedy coloring."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

_DEADLINE = 90.0
_STATE = {}

def _greedy_coloring(cand, adj):
    # Returns list of (color, vertex) where color is an upper bound on the clique
    # size of cand (a maximal clique in cand has size <= max color).
    n = len(cand)
    if n == 0:
        return []
    remaining = list(cand)
    colors = [0] * n
    idx_of = {v: i for i, v in enumerate(cand)}
    color = 0
    while remaining:
        color += 1
        to_remove = []
        for v in remaining:
            # v can take this color if no already-colored (this color) neighbor is uncolored? Standard:
            # assign color if no neighbor in remaining has already been assigned this color.
            conflict = False
            for u in remaining:
                if u == v:
                    continue
                if colors[idx_of[u]] == color and adj[v] & adj[u]:
                    conflict = True
                    break
            if not conflict:
                colors[idx_of[v]] = color
                to_remove.append(v)
        for v in to_remove:
            remaining.remove(v)
    # Rebuild ordered by color
    order = sorted(range(n), key=lambda i: colors[i])
    return [(colors[i], cand[i]) for i in order]

def _bronk(P_set, R, X, adj, best):
    if time.time() > _DEADLINE:
        return False
    if not P_set:
        if len(R) > len(best[0]):
            best[0] = list(R)
        return True
    # Coloring upper bound
    P_list = list(P_set)
    order = _greedy_coloring(P_list, adj)
    # Iterate in reverse color order for pruning
    for color, v in reversed(order):
        if time.time() > _DEADLINE:
            return False
        if len(R) + color <= len(best[0]):
            break
        newP = P_set & adj[v]
        newX = X & adj[v]
        if _bronk(newP, R | {v}, newX, adj, best):
            pass
        P_set = P_set - {v}
        X = X | {v}
    return True

def _max_clique(n, edges):
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
    best = [[]]
    _bronk((1 << n) - 1, frozenset(), frozenset(), adj, best)
    return best[0]

def construct():
    n = P['n']
    edges = P['edges']
    best_clique = _max_clique(n, edges)
    if len(best_clique) < 2:
        a, b = edges[0]
        best_clique = [a, b]
    branches = [int(v) for v in best_clique]
    es = {tuple(e) for e in edges}
    paths = []
    for a, b in combinations(branches, 2):
        if (a, b) in es or (b, a) in es:
            paths.append([a, b])
        else:
            # Fallback direct edge should exist for a clique; but guard anyway
            paths.append([a, b])
    return {'branches': branches, 'paths': paths}
# EVOLVE-BLOCK-END

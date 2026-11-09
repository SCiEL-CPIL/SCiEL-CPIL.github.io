"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    import time
    from itertools import combinations
    deadline = time.time() + 55  # safety margin before 60s kill

    n = P['n']
    adj = [set() for _ in range(n)]
    es = set()
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        es.add((u, v))
        es.add((v, u))

    def is_clique(vertices):
        for a, b in combinations(vertices, 2):
            if b not in adj[a]:
                return False
        return True

    # Bron-Kerbosch with pivot to find a maximum clique
    best = []

    def bk(R, Pset, X):
        nonlocal best
        if time.time() > deadline:
            return
        if not Pset and not X:
            if len(R) > len(best):
                best = list(R)
            return
        # choose pivot with max |N(u) & P|
        Plist = list(Pset)
        pivot = max(Plist, key=lambda u: len(adj[u] & Pset)) if Plist else None
        candidates = [v for v in Plist if (pivot is None or v not in adj[pivot])]
        for v in candidates:
            if time.time() > deadline:
                return
            R.add(v)
            P_new = Pset & adj[v]
            X_new = X & adj[v]
            bk(R, P_new, X_new)
            R.discard(v)
            Pset.discard(v)
            X.add(v)

    bk(set(), set(range(n)), set())

    # Fallback if nothing found
    if len(best) < 2:
        best = [P['edges'][0][0], P['edges'][0][1]]

    # Build result: branches = clique, paths = direct edges between each pair
    branches = [int(x) for x in best]
    paths = []
    for a, b in combinations(branches, 2):
        # direct edge path [a, b]; interiors empty (disjoint, avoid branches)
        paths.append([int(a), int(b)])

    return {'branches': branches, 'paths': paths}
# EVOLVE-BLOCK-END

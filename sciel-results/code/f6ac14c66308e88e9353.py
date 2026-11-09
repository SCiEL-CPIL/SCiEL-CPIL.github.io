"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _seed_biclique(adj, L, R, n, deadline):
    # Greedily grow R = common-neighborhood of L, then L = common-neighborhood of R.
    # Stop once no further vertex can be added or the deadline is near.
    L = set(L)
    R = set(R)
    while True:
        if time.time() >= deadline:
            break
        if not L or not R:
            break
        # candidate R: neighbors of every L vertex
        newR = set(adj[L.pop()])
        L.add(L)  # placeholder, recompute below
        # recompute properly
        Lset = set(L)
        newR = set(adj[next(iter(Lset))])
        for v in Lset:
            newR &= adj[v]
        newR -= Lset
        # candidate L: neighbors of every R vertex
        Rset = set(R)
        newL = set(adj[next(iter(Rset))])
        for v in Rset:
            newL &= adj[v]
        newL -= Rset
        if len(newR) == len(R) and len(newL) == len(Lset):
            break
        R = newR
        L = newL
    return L, R

def _try_expand(adj, L, R, n, deadline):
    # Attempt to add one vertex to L or R while keeping the biclique valid.
    # Returns (L, R, improved)
    Lset = set(L)
    Rset = set(R)
    # Candidate to add to L: neighbor of all in R, not in R
    candL = set(adj[next(iter(Rset))])
    for v in Rset:
        candL &= adj[v]
    candL -= Rset
    if candL:
        # pick the candidate that maximizes potential (degree within candL neighbors)
        best = max(candL, key=lambda v: len(adj[v] & candL))
        Lset.add(best)
        # After adding to L, R must shrink to common neighbors
        newR = set(adj[next(iter(Lset))])
        for v in Lset:
            newR &= adj[v]
        newR -= Lset
        return Lset, newR, True
    # Candidate to add to R: neighbor of all in L, not in L
    candR = set(adj[next(iter(Lset))])
    for v in Lset:
        candR &= adj[v]
    candR -= Lset
    if candR:
        best = max(candR, key=lambda v: len(adj[v] & candR))
        Rset.add(best)
        newL = set(adj[next(iter(Rset))])
        for v in Rset:
            newL &= adj[v]
        newL -= Rset
        return newL, Rset, True
    return Lset, Rset, False

def _find_biclique(adj, n, deadline):
    # Find the largest biclique in the current residual graph.
    # Strategy: for each vertex u, seed with its neighbors, expand, then try
    # to grow by single additions. Keep the best found.
    best = (0, None, None)  # (size, L, R)
    for u in range(n):
        if time.time() >= deadline:
            break
        neigh = adj[u]
        if not neigh:
            continue
        # Seed: L={u}, R=neigh(u), then expand
        L = {u}
        R = set(neigh)
        # Expand to closure
        L, R = _seed_biclique(adj, L, R, n, deadline)
        size = len(L) * len(R)
        if size > best[0]:
            best = (size, set(L), set(R))
        # Try a few single-expansion steps from this seed
        for _ in range(3):
            if time.time() >= deadline:
                break
            L2, R2, imp = _try_expand(adj, L, R, n, deadline)
            if not imp:
                break
            L, R = L2, R2
            size = len(L) * len(R)
            if size > best[0]:
                best = (size, set(L), set(R))
    return best[1], best[2]

def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin under 90s
    n = P['n']
    adj = _build_adj()
    bicliques = []
    # Track which edges are covered
    covered = set()
    for a, b in P['edges']:
        covered.add((a, b))
    remaining = set(covered)
    while remaining:
        if time.time() >= deadline:
            break
        L, R = _find_biclique(adj, n, deadline)
        if not L or not R:
            # Fallback: take a single edge
            a, b = next(iter(remaining))
            L = {a}
            R = {b}
        # Remove edges covered by this biclique
        for u in L:
            for v in R:
                if u != v:
                    key = (min(u, v), max(u, v))
                    if key in remaining:
                        remaining.discard(key)
        bicliques.append([sorted(L), sorted(R)])
        if time.time() >= deadline:
            break
    # If we ran out of time with edges remaining, cover them one-by-one
    if remaining:
        for a, b in sorted(remaining):
            bicliques.append([[a], [b]])
        remaining = set()
    # Convert to plain Python ints and lists
    result = []
    for L, R in bicliques:
        result.append([[int(x) for x in L], [int(x) for x in R]])
    return result
# EVOLVE-BLOCK-END

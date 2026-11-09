"""Biclique partition of a random graph – iterative max-biclique extraction."""
import time
import itertools

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _build_adj(n, edges):
    """Return adjacency as a set of frozensets for fast membership and a list of edge tuples."""
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(frozenset((u, v)))
    return adj, edge_set


def _expand_biclique(L, R, adj, active, deadline):
    """Iteratively expand biclique (L, R) by adding vertices to either side."""
    L = set(L)
    R = set(R)
    # Ensure consistency: R = intersection of N(l) for l in L
    if L:
        new_R = set(adj[list(L)[0]])
        for v in L:
            new_R &= adj[v]
        R = new_R
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try to add vertices to L
        for w in active:
            if time.time() > deadline:
                break
            if w not in L and R.issubset(adj[w]):
                L.add(w)
                changed = True
        # Shrink R to intersection of N(l) for l in L
        if L:
            new_R = set(adj[list(L)[0]])
            for v in L:
                new_R &= adj[v]
            if new_R != R:
                R = new_R
                changed = True
    # Try to add vertices to R
    if L:
        for r in active:
            if time.time() > deadline:
                break
            if r not in R and all(r in adj[l] for l in L):
                R.add(r)
                changed = True
    return L, R


def _find_max_biclique(adj, n, edge_set, deadline):
    """
    Heuristic search for a large biclique in the current residual graph.
    Tries multiple seed strategies and expands greedily.
    Returns (left, right, edge_count) for the best biclique found.
    """
    best_left = None
    best_right = None
    best_count = 0

    active = [v for v in range(n) if adj[v]]
    if not active:
        return None, None, 0

    def try_expand(L_init, R_init):
        nonlocal best_left, best_right, best_count
        if time.time() > deadline:
            return
        L_set, R_set = _expand_biclique(L_init, R_init, adj, active, deadline)
        if not L_set or not R_set:
            return
        count = len(L_set) * len(R_set)
        if count > best_count:
            best_count = count
            best_left = sorted(L_set)
            best_right = sorted(R_set)

    # Strategy 1: Single vertex seeds on left
    for u in active:
        if time.time() > deadline:
            break
        try_expand({u}, set(adj[u]))

    # Strategy 2: Edge seeds - u on left, v on right
    for u in active:
        if time.time() > deadline:
            break
        for v in adj[u]:
            if time.time() > deadline:
                break
            # L={u}, R={v} then expand
            try_expand({u}, {v})

    # Strategy 3: Two vertices on left
    for i, u in enumerate(active):
        if time.time() > deadline:
            break
        for v in active[i+1:]:
            if time.time() > deadline:
                break
            R_init = adj[u] & adj[v]
            if R_init:
                try_expand({u, v}, R_init)

    # Strategy 4: Two vertices, one on each side, with common neighbors
    for u in active:
        if time.time() > deadline:
            break
        for v in list(adj[u])[:10]:  # Limit to avoid O(n^2)
            if time.time() > deadline:
                break
            # L={u}, R={v}, but also try L={u}, R=N(u)∩{neighbors that connect to u}
            # Already covered in strategy 1
            pass

    if best_left is None or best_right is None or best_count == 0:
        return None, None, 0

    return best_left, best_right, best_count


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = P['n']
    adj, edge_set = _build_adj(n, P['edges'])

    bicliques = []
    remaining_edges = set(edge_set)

    while remaining_edges:
        if time.time() > deadline:
            break

        left, right, count = _find_max_biclique(adj, n, edge_set, deadline)

        if left is None or right is None or count == 0:
            # Fallback: pick any remaining edge as a 1x1 biclique
            e = next(iter(remaining_edges))
            u, v = tuple(e)
            left = [u]
            right = [v]
            count = 1
        else:
            # Verify that all cross pairs are actually edges (they should be by construction)
            # Remove covered edges from remaining
            covered = set()
            for u in left:
                for v in right:
                    fs = frozenset((u, v))
                    if fs in remaining_edges:
                        covered.add(fs)

            if not covered:
                # Shouldn't happen, but safety
                e = next(iter(remaining_edges))
                u, v = tuple(e)
                left = [u]
                right = [v]
                covered = {frozenset((u, v))}

            remaining_edges -= covered

        # Update adjacency for residual
        for u in left:
            for v in right:
                if v in adj[u]:
                    adj[u].discard(v)
                if u in adj[v]:
                    adj[v].discard(u)

        bicliques.append([list(left), list(right)])

    # Fallback: if we ran out of time with edges remaining, cover them as single edges
    if remaining_edges and time.time() < deadline:
        for e in list(remaining_edges):
            if time.time() > deadline:
                break
            u, v = tuple(e)
            bicliques.append([[u], [v]])
            remaining_edges.discard(e)

    # Ensure types are correct: Python int, list
    result = []
    for bic in bicliques:
        L = [int(x) for x in bic[0]]
        R = [int(x) for x in bic[1]]
        result.append([L, R])

    return result
# EVOLVE-BLOCK-END

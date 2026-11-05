"""Clique partition of two fixed dense random graphs via greedy extraction + refinement."""
import time

_DEADLINE = 110.0  # safety margin under 150s wall-clock kill
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _make_graph(n, p, seed):
    from random import Random
    rng = Random(seed)
    adj = [[False] * n for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u][v] = True
                adj[v][u] = True
    return adj


def _neighbors(adj, v):
    return [u for u in range(len(adj)) if adj[v][u]]


def _max_clique(adj, candidates):
    """Bron–Kerbosch (non-recursive) returning the largest clique among candidates."""
    best = []
    stack = [(list(candidates), [], [])]
    while stack:
        cand, path, excl = stack.pop()
        if len(path) + len(cand) <= len(best):
            continue
        if not cand:
            if len(path) > len(best):
                best = path[:]
            continue
        # pivot: vertex in cand with most neighbors in cand
        pivot = max(cand, key=lambda x: sum(1 for y in cand if y != x and adj[x][y]))
        for v in cand:
            if adj[pivot][v]:
                continue
            new_cand = [u for u in cand if u > v and adj[v][u]]
            stack.append((new_cand, path + [v], excl + [u for u in cand if u < v]))
    return best


def _greedy_clique_partition(adj, n, time_budget):
    """Extract large cliques greedily; return list of cliques (each a list of vertices)."""
    remaining = set()
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                remaining.add((u, v))

    cliques = []
    while remaining:
        if _time_left() < time_budget:
            break
        # pick an edge with the smallest number of common neighbors (hardest to place)
        u, v = min(remaining, key=lambda e: len(_common(adj, e[0], e[1])))
        cand = _common(adj, u, v)
        clique = _max_clique(adj, cand)
        if len(clique) < 2:
            clique = [u, v]
        else:
            # ensure u and v are in the clique
            if u not in clique:
                clique.append(u)
            if v not in clique:
                clique.append(v)
        # remove all edges within the clique
        k = len(clique)
        for i in range(k):
            for j in range(i + 1, k):
                a, b = clique[i], clique[j]
                e = (a, b) if a < b else (b, a)
                remaining.discard(e)
        cliques.append(clique)
    return cliques, remaining


def _common(adj, u, v):
    return [w for w in range(len(adj)) if w != u and w != v and adj[u][w] and adj[v][w]]


def _refine(adj, n, cliques, remaining, time_budget):
    """Merge a clique with an adjacent vertex if that vertex's remaining edges fit inside."""
    # index edges
    edge_to_clique = {}
    for idx, c in enumerate(cliques):
        k = len(c)
        for i in range(k):
            for j in range(i + 1, k):
                a, b = c[i], c[j]
                e = (a, b) if a < b else (b, a)
                edge_to_clique[e] = idx

    changed = True
    while changed:
        if _time_left() < time_budget:
            break
        changed = False
        # For each clique, check if any neighbor vertex can be absorbed
        for idx in range(len(cliques)):
            c = cliques[idx]
            cset = set(c)
            # find a vertex v not in c that is adjacent to all of c
            for v in range(n):
                if v in cset:
                    continue
                if all(adj[v][w] for w in c):
                    # check all remaining edges of v are within c
                    ok = True
                    for w in range(n):
                        if w == v or w in cset:
                            continue
                        if adj[v][w]:
                            e = (v, w) if v < w else (w, v)
                            # this edge must be in the same clique
                            if e in edge_to_clique and edge_to_clique[e] == idx:
                                continue
                            ok = False
                            break
                    if ok:
                        # absorb v
                        cliques[idx] = c + [v]
                        cset.add(v)
                        # update edge_to_clique for new edges
                        for w in c:
                            e = (v, w) if v < w else (w, v)
                            edge_to_clique[e] = idx
                        changed = True
                        break
    return cliques


def _solve_instance(n, p, seed, time_budget):
    adj = _make_graph(n, p, seed)
    cliques, remaining = _greedy_clique_partition(adj, n, time_budget)

    # If greedy left edges (shouldn't happen if we processed all), add them as 2-cliques
    if remaining:
        for e in sorted(remaining):
            cliques.append(list(e))

    # Refinement pass
    cliques = _refine(adj, n, cliques, remaining, time_budget)
    return cliques


def construct():
    t0 = time.time()
    total_budget = _DEADLINE - (t0 - _START)
    # Split budget between two instances
    budget_each = total_budget / 2.0
    results = []
    for i, (n, p, seed) in enumerate(((32, 0.6, 601), (64, 0.55, 602))):
        b = budget_each if i == 0 else _time_left()
        cliques = _solve_instance(n, p, seed, max(1.0, b))
        results.append(cliques)
    return results

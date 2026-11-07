"""Rewritten program for split-graph clique partition with minimum cliques."""

def construct():
    import time
    from random import Random
    from itertools import combinations

    _start = time.time()
    _DEADLINE = 115.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set()
        for e in edges:
            a, b = e
            if a > b:
                a, b = b, a
            edge_set.add((a, b))

        cross_adj = {}
        for v in range(k, n):
            nbrs = set()
            for u in range(k):
                if (u, v) in edge_set:
                    nbrs.add(u)
            cross_adj[v] = nbrs

        cliques = []
        covered_C_edges = set()

        for v in range(k, n):
            nbrs = cross_adj[v]
            if len(nbrs) >= 1:
                clique = [v] + sorted(nbrs)
                cliques.append(clique)
                for a, b in combinations(sorted(nbrs), 2):
                    covered_C_edges.add((a, b))

        remaining = set()
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) not in covered_C_edges:
                    remaining.add((u, v))

        if not remaining:
            return cliques

        rem_adj = [set() for _ in range(k)]
        for (a, b) in remaining:
            rem_adj[a].add(b)
            rem_adj[b].add(a)

        def bronk_max_clique(adj, n):
            best = set()

            def bronk(R, P, X):
                nonlocal best
                if not P and not X:
                    if len(R) > len(best):
                        best = set(R)
                    return
                if P:
                    u = max(P | X, key=lambda x: len(P & adj[x]))
                    candidates = list(P - adj[u])
                else:
                    candidates = []
                for v in candidates:
                    bronk(R | {v}, P & adj[v], X & adj[v])
                    P.discard(v)
                    X.add(v)

            bronk(set(), set(range(n)), set())
            return best

        while remaining:
            if time.time() - _start > _DEADLINE - 3:
                for (a, b) in remaining:
                    cliques.append([a, b])
                break

            mc = bronk_max_clique(rem_adj, k)
            if len(mc) < 2:
                for (a, b) in remaining:
                    cliques.append([a, b])
                break

            cliques.append(sorted(mc))
            for a, b in combinations(sorted(mc), 2):
                remaining.discard((a, b))
                rem_adj[a].discard(b)
                rem_adj[b].discard(a)

        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() - _start > _DEADLINE - 3:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            results.append(solve(*spec))

    return results

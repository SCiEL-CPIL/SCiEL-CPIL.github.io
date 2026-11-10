def construct():
    import time
    _start = time.time()
    _DEADLINE = 140.0

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def ekey(a, b):
        return (min(a, b), max(a, b))

    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        C = list(range(k))
        I = list(range(k, n))

        cross = {i: set() for i in I}
        C_edges = set()
        for (a, b) in edges:
            if a < k and b < k:
                C_edges.add(ekey(a, b))
            else:
                if a < k:
                    c, i = a, b
                else:
                    c, i = b, a
                cross[i].add(c)

        claimed = set()
        cliques = []

        # unclaimed adjacency on C
        unclaimed_adj = {v: set() for v in C}
        for (a, b) in C_edges:
            unclaimed_adj[a].add(b)
            unclaimed_adj[b].add(a)

        def mark_claimed(group):
            for x in range(len(group)):
                for y in range(x + 1, len(group)):
                    e = ekey(group[x], group[y])
                    if e in C_edges and e not in claimed:
                        claimed.add(e)
                        unclaimed_adj[group[x]].discard(group[y])
                        unclaimed_adj[group[y]].discard(group[x])

        # Process I vertices
        for i in I:
            N = list(cross[i])
            if not N:
                continue
            remaining = set(N)
            groups = []
            while remaining:
                if time.time() > _DEADLINE - 3:
                    # fallback: singletons
                    for v in remaining:
                        groups.append([v])
                    remaining = set()
                    break
                # find a maximal clique in unclaimed subgraph induced on remaining
                # start with a vertex, grow
                start = min(remaining)
                clique = {start}
                cand = unclaimed_adj[start] & remaining
                for v in sorted(cand):
                    if all(v in unclaimed_adj[u] for u in clique):
                        clique.add(v)
                groups.append(sorted(clique))
                mark_claimed(clique)
                remaining -= clique
            for g in groups:
                clique_list = [i] + g
                if len(clique_list) >= 2:
                    cliques.append(clique_list)

        # Cover remaining unclaimed C-C edges with C-only cliques
        rem_edges = set()
        for v in C:
            for w in unclaimed_adj[v]:
                if v < w:
                    rem_edges.add(ekey(v, w))
        while rem_edges:
            if time.time() > _DEADLINE - 3:
                for (a, b) in rem_edges:
                    cliques.append([a, b])
                break
            a, b = next(iter(rem_edges))
            clique = {a, b}
            cand = unclaimed_adj[a] & unclaimed_adj[b]
            for v in sorted(cand):
                if all(v in unclaimed_adj[u] for u in clique):
                    clique.add(v)
            clique_list = sorted(clique)
            cliques.append(clique_list)
            cl = clique_list
            for x in range(len(cl)):
                for y in range(x + 1, len(cl)):
                    e = ekey(cl[x], cl[y])
                    rem_edges.discard(e)
                    unclaimed_adj[cl[x]].discard(cl[y])
                    unclaimed_adj[cl[y]].discard(cl[x])

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > _DEADLINE - 3:
            results.append([[list(e) for e in instance(*spec)]])
            continue
        results.append(solve(*spec))
    return results

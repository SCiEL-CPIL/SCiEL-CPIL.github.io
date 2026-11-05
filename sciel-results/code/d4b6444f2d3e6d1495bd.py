def construct():
    import time
    start_time = time.time()
    DEADLINE = 130

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve(edges, n):
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))

        uncovered = set(edge_set)
        cliques = []

        while uncovered:
            if time.time() - start_time > DEADLINE:
                for (u, v) in uncovered:
                    cliques.append([u, v])
                break

            active = set()
            for (u, v) in uncovered:
                active.add(u)
                active.add(v)

            clique = set()
            remaining = set(active)
            while remaining:
                best = None
                best_deg = -1
                for v in remaining:
                    ok = True
                    for w in clique:
                        if (min(v, w), max(v, w)) not in uncovered:
                            ok = False
                            break
                    if not ok:
                        continue
                    deg = 0
                    for x in remaining:
                        if x != v and (min(v, x), max(v, x)) in uncovered:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best = v
                if best is None:
                    break
                clique.add(best)
                remaining.discard(best)

            if len(clique) < 2:
                (u, v) = next(iter(uncovered))
                clique = {u, v}

            cliques.append(list(clique))
            for a in clique:
                for b in clique:
                    if a < b:
                        uncovered.discard((a, b))

        return cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = instance(n, p, seed)
        cliques = solve(edges, n)
        results.append(cliques)

    return results

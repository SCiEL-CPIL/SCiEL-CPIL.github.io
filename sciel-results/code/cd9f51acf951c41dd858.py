import time

def construct():
    start = time.time()
    DEADLINE = 140.0  # safety margin before 150s

    def build_graph(n, p, seed):
        from random import Random
        rng = Random(seed)
        adj = [0]*n
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def max_clique(adj, n):
        # Tomita maximum clique with greedy coloring, on graph adj (bitmasks)
        best = []
        best_size = 0

        def color_sort(P):
            # greedy coloring, returns list of (vertex, color)
            order = []
            colors = {}
            remaining = P
            color_idx = 0
            while remaining:
                color_idx += 1
                # pick a maximal independent set in remaining for this color
                # greedy: take vertices in order
                cset = []
                for v in list(remaining):
                    if v in remaining:
                        # check v independent from cset
                        ok = True
                        for u in cset:
                            if (adj[v] >> u) & 1:
                                ok = False
                                break
                        if ok:
                            cset.append(v)
                            remaining.discard(v)
                for v in cset:
                    colors[v] = color_idx
            # order by color
            order = sorted(colors.keys(), key=lambda x: colors[x])
            return order, colors

        def expand(R, P):
            nonlocal best, best_size
            if not P:
                if len(R) > best_size:
                    best_size = len(R)
                    best = list(R)
                return
            if len(R) + len(P) <= best_size:
                return
            order, colors = color_sort(P)
            # process in reverse color order
            for v in reversed(order):
                if len(R) + colors[v] <= best_size:
                    return
                # branch: include v
                R.append(v)
                P_new = [u for u in P if (adj[v] >> u) & 1]
                expand(R, P_new)
                R.pop()
                P.remove(v)

        expand([], list(range(n)))
        return best

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE:
            # fallback: vertex cover via greedy
            adj = build_graph(n, p, seed)
            # greedy vertex cover
            covered = set()
            cover = set()
            edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if (adj[u] >> v) & 1:
                        edges.append((u, v))
            # greedy: pick vertex covering most uncovered edges
            while edges:
                best_v = -1
                best_cnt = -1
                for v in range(n):
                    cnt = sum(1 for (a,b) in edges if a==v or b==v)
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_v = v
                if best_cnt <= 0:
                    break
                cover.add(best_v)
                edges = [(a,b) for (a,b) in edges if a!=best_v and b!=best_v]
            results.append(sorted(cover))
            continue
        adj = build_graph(n, p, seed)
        # complement graph
        comp = [0]*n
        for u in range(n):
            full = ((1 << n) - 1) ^ (1 << u)
            comp[u] = full & ~adj[u]
        clique = max_clique(comp, n)
        # clique in complement = independent set in G
        # min vertex cover = all vertices - max independent set
        independent = set(clique)
        cover = [v for v in range(n) if v not in independent]
        results.append(sorted(cover))
    return results

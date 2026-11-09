import time

def construct():
    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, n):
        remaining = set()
        for (u, v) in edges:
            remaining.add((min(u, v), max(u, v)))

        pieces = []
        deadline = time.time() + 115  # safety margin within 120s

        def build_cur_adj():
            cur_adj = {i: [] for i in range(n)}
            for (u, v) in remaining:
                cur_adj[u].append(v)
                cur_adj[v].append(u)
            for i in range(n):
                cur_adj[i].sort()
            return cur_adj

        def find_longest_cycle():
            cur_adj = build_cur_adj()
            best_cycle = None
            best_len = 0
            for start in range(n):
                if not cur_adj[start]:
                    continue
                visited = [False] * n
                in_stack = [False] * n
                depth = [-1] * n
                parent = [-1] * n
                cycles_found = []

                def dfs(u):
                    visited[u] = True
                    in_stack[u] = True
                    for v in cur_adj[u]:
                        if not visited[v]:
                            parent[v] = u
                            depth[v] = depth[u] + 1
                            dfs(v)
                        elif in_stack[v]:
                            if depth[u] - depth[v] >= 2:
                                # reconstruct cycle v -> ... -> u
                                cycle = []
                                cur = u
                                while cur != v:
                                    cycle.append(cur)
                                    cur = parent[cur]
                                cycle.append(v)
                                cycle.reverse()
                                cycles_found.append(cycle)
                    in_stack[u] = False

                depth[start] = 0
                dfs(start)
                for cyc in cycles_found:
                    L = len(cyc)
                    if L > best_len:
                        best_len = L
                        best_cycle = cyc
            return best_cycle

        while True:
            if time.time() > deadline:
                break
            cyc = find_longest_cycle()
            if cyc is None:
                break
            # validate
            k = len(cyc)
            if k < 3 or len(set(cyc)) != k:
                # invalid, shouldn't happen; break to avoid infinite loop
                break
            ok = True
            for i in range(k):
                a = cyc[i]
                b = cyc[(i + 1) % k]
                e = (min(a, b), max(a, b))
                if e not in remaining:
                    ok = False
                    break
            if not ok:
                break
            for i in range(k):
                a = cyc[i]
                b = cyc[(i + 1) % k]
                e = (min(a, b), max(a, b))
                remaining.discard(e)
            pieces.append(list(cyc))

        for (u, v) in sorted(remaining):
            pieces.append([u, v])

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n = spec[0]
        edges = instance(*spec)
        results.append(decompose(edges, n))
    return results

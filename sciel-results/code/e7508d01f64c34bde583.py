"""Deterministic cycle-avoiding greedy builder for small k-regular girth-g graphs."""
# EVOLVE-BLOCK-START
def construct(k, g):
    from collections import deque

    adj = []

    def add_vertex():
        adj.append([])
        return len(adj) - 1

    def add_edge(u, v):
        adj[u].append(v)
        adj[v].append(u)

    def degree(v):
        return len(adj[v])

    def same_component(u, v):
        seen = {u}
        dq = deque([u])
        while dq:
            x = dq.popleft()
            if x == v:
                return True
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    dq.append(y)
        return False

    def shortest_nonbacktracking(u, v):
        if u == v:
            return 0
        start = (u, -1)
        dist = {start: 0}
        dq = deque([start])
        while dq:
            x, prev = dq.popleft()
            d = dist[(x, prev)]
            for y in adj[x]:
                if y == prev:
                    continue
                st = (y, x)
                if st in dist:
                    continue
                dist[st] = d + 1
                if y == v:
                    return d + 1
                dq.append(st)
        return None

    def would_create_short_cycle(u, v):
        if not same_component(u, v):
            return False
        L = shortest_nonbacktracking(u, v)
        if L is None:
            return False
        return (L + 1) < g

    add_vertex()
    max_iter = 500000
    it = 0
    while it < max_iter:
        it += 1
        u = -1
        for v in range(len(adj)):
            if degree(v) < k:
                u = v
                break
        if u == -1:
            break
        done = False
        for v in range(len(adj)):
            if v == u or degree(v) >= k:
                continue
            if v in adj[u]:
                continue
            if would_create_short_cycle(u, v):
                continue
            add_edge(u, v)
            done = True
            break
        if not done:
            w = add_vertex()
            add_edge(u, w)

    edges = []
    for u in range(len(adj)):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return len(adj), edges
# EVOLVE-BLOCK-END

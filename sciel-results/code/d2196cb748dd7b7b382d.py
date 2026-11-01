def construct(k, g):
    import time
    from collections import deque
    start_time = time.time()
    deadline = start_time + 115

    def build_luw(q, d):
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = (x[c] + y[a] * x[b]) % q
                    else:
                        y[c] = (x[c] - x[a] * y[b]) % q
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def has_short_cycle(n, edges, target):
        """Exact: True iff a cycle of length < target exists."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        max_depth = (target - 1) // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            dq = deque([s])
            while dq:
                u = dq.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        dq.append(v)
                    elif v != parent[u] and parent[v] != u and dist[u] + dist[v] + 1 < target:
                        return True
        return False

    def is_regular_simple(n, edges, k):
        deg = [0] * n
        seen = set()
        for u, v in edges:
            if u == v:
                return False
            key = (u, v) if u < v else (v, u)
            if key in seen:
                return False
            seen.add(key)
            deg[u] += 1
            deg[v] += 1
        for d in deg:
            if d != k:
                return False
        return True

    best = None
    for d in range(2, 13):
        if time.time() > deadline:
            break
        try:
            n, edges = build_luw(k, d)
            if not is_regular_simple(n, edges, k):
                continue
            if not has_short_cycle(n, edges, g):
                if best is None or n < best[0]:
                    best = (n, edges)
                return n, edges
        except Exception:
            continue

    if best is not None:
        return best

    # Fallback: a regular LUW graph (best effort)
    d = max(4, (g - 2) // 2)
    n, edges = build_luw(k, d)
    return n, edges

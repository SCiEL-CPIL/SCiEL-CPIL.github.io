import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    def _luw_build(q, d):
        """Build LUW graph with parameters q (field size) and d (depth)."""
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
                    y[c] = (x[c] + y[a] * x[b]) % q if side == 0 else (x[c] - x[a] * y[b]) % q
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

    def _verify_girth(n, edges, target_g):
        """Verify graph is k-regular, simple, and has girth >= target_g."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            if u == v:
                return False
            adj[u].append(v)
            adj[v].append(u)

        for i in range(n):
            if len(adj[i]) != k:
                return False

        seen = set()
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key in seen:
                return False
            seen.add(key)

        for start in range(n):
            dist = [-1] * n
            prev = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                du = dist[u]
                if du >= target_g - 1:
                    break
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        prev[v] = u
                        q.append(v)
                    elif v != prev[u]:
                        cycle_len = du + dist[v] + 1
                        if cycle_len < target_g:
                            return False
        return True

    # Systematically try (q, d) pairs for LUW construction
    best_n = float('inf')
    best_edges = None

    min_d = max(3, (g - 2) // 2)
    if (g - 2) % 2 != 0:
        min_d += 1

    for d in range(min_d, min_d + 6):
        if time.time() > deadline - 15:
            break
        for q in range(2, 8):
            if time.time() > deadline - 15:
                break

            est_n = 2 * (q ** d)
            if est_n >= best_n:
                continue

            try:
                n, edges = _luw_build(q, d)
            except Exception:
                continue

            if n >= best_n:
                continue

            if _verify_girth(n, edges, g):
                best_n = n
                best_edges = edges

        if time.time() > deadline - 15:
            break

    if best_edges is not None:
        return best_n, best_edges

    # Fallback: original LUW parameters
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    q = 5
    try:
        n, edges = _luw_build(q, d)
        if _verify_girth(n, edges, g):
            return n, edges
    except Exception:
        pass

    # Last resort
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    q = 5
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
                y[c] = (x[c] + y[a] * x[b]) % q if side == 0 else (x[c] - x[a] * y[b]) % q
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

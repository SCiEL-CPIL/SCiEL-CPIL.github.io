import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140

    if k != 3 or g != 16:
        return _luw_construct(k, g)

    # Triangular prism base graph: 6 vertices, 9 edges, 3-regular
    base_edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]
    base_vertices = list(range(6))
    adj = {v: [] for v in base_vertices}
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)

    def check_girth(voltages, n):
        """Check if no non-backtracking closed walk of length < 16 has identity voltage."""
        def get_vol(u, v):
            if u < v:
                return voltages[(u, v)]
            else:
                return (-voltages[(v, u)]) % n

        for start in base_vertices:
            for first in adj[start]:
                v1 = get_vol(start, first)
                for second in adj[first]:
                    if second == start:
                        continue
                    v2 = (v1 + get_vol(first, second)) % n
                    stack = [(second, start, v2, 2)]
                    while stack:
                        cur, prev, vprod, L = stack.pop()
                        if cur == start and vprod == 0:
                            return False
                        if L >= 15:
                            continue
                        for nxt in adj[cur]:
                            if nxt == prev:
                                continue
                            nv = (vprod + get_vol(cur, nxt)) % n
                            stack.append((nxt, cur, nv, L + 1))
        return True

    for n in range(156, 300):
        if time.time() > deadline:
            break
        for trial in range(5000):
            if time.time() > deadline:
                break
            voltages = {}
            for u, v in base_edges:
                if u < v:
                    voltages[(u, v)] = random.randint(0, n - 1)
            if check_girth(voltages, n):
                edges = []
                for u, v in base_edges:
                    vol = voltages[(min(u, v), max(u, v))] if u < v else (-voltages[(v, u)]) % n
                    for gi in range(n):
                        edges.append((u * n + gi, v * n + (gi + vol) % n))
                # Verify simple graph
                seen = set()
                for a, b in edges:
                    if a == b:
                        break
                    key = (min(a, b), max(a, b))
                    if key in seen:
                        break
                    seen.add(key)
                else:
                    return 6 * n, edges
    return _luw_construct(k, g)


def _luw_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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

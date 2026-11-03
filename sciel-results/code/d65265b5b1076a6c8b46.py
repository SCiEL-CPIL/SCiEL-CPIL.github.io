"""LUW CD(d,q) cage construction with girth-verified depth search."""
import time

def _luw_build(k, d):
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: table[x][y])
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
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
    index, order, adj = {start: 0}, [start], []
    for vertex in order:
        idx = index[vertex]
        adj.append([])
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                adj.append([])
            adj[idx].append(index[other])
    for i in range(len(order)):
        for j in adj[i]:
            adj[j].append(i)
    return len(order), adj


def _verify_girth(adj, g):
    n = len(adj)
    for s in range(n):
        dist = [-1] * n
        par = [-1] * n
        dist[s] = 0
        q = [s]
        for v in q:
            nd = dist[v] + 1
            if nd >= g:
                return False
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = nd
                    par[w] = v
                    q.append(w)
                elif w != par[v]:
                    return False
    return True


def construct(k, g):
    t0 = time.time()
    deadline = t0 + 115.0

    def build(d):
        return _luw_build(k, d)

    # try small depths first; keep the smallest whose girth is verified >= g
    best = None
    for d in range(3, 20):
        if time.time() > deadline:
            break
        n, adj = build(d)
        if _verify_girth(adj, g):
            best = (n, adj)
            break
    # fallback: depth guaranteed by the LUW bound (girth >= d+5)
    if best is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        best = build(d)
    n, adj = best
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    return n, edges

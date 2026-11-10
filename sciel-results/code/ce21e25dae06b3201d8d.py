"""Construct a k-regular graph of girth g with minimum vertices."""
import time
import random
from collections import deque

def _verify_girth(adj, g):
    """Verify girth >= g using non-backtracking walks from every vertex."""
    n = len(adj)
    for start in range(n):
        min_depth = {}
        queue = deque()
        for nb in adj[start]:
            state = (nb, start)
            min_depth[state] = 1
            queue.append(state)
        while queue:
            node, prev = queue.popleft()
            depth = min_depth[(node, prev)]
            if depth >= g - 1:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == start:
                    if depth + 1 < g:
                        return False
                    continue
                new_state = (nb, node)
                if new_state not in min_depth:
                    min_depth[new_state] = depth + 1
                    queue.append(new_state)
    return True

def _construct_luw(k, g):
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

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    # Get baseline from LUW
    best_n, best_edges = _construct_luw(k, g)

    # Try random search for smaller graphs
    rng = random.Random(12345)
    half = (g - 1) // 2
    moore = 2 * sum(k**i for i in range(half + 1))
    min_n = max(moore, 50)

    for n in range(best_n - 1, min_n - 1, -1):
        if time.time() > deadline - 30:
            break
        attempts = 2000
        for _ in range(attempts):
            if time.time() > deadline - 30:
                break
            # Generate random k-regular graph
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            rng.shuffle(stubs)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if not valid:
                continue
            if _verify_girth(adj, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges

    return best_n, best_edges

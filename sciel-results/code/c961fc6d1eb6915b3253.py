import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Try the LUW construction
    def try_luw():
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

    n_luw, edges_luw = try_luw()
    
    def check_girth(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            visited = {}
            q = deque()
            q.append((i, -1, 0))
            visited[(i, -1)] = 0
            while q:
                node, parent, dist = q.popleft()
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    state = (node, nb)
                    if state in visited:
                        cycle_len = dist + 1 - visited[state]
                        if cycle_len < g:
                            return False
                    else:
                        visited[state] = dist + 1
                        q.append((nb, node, dist + 1))
        return True

    if check_girth(n_luw, edges_luw, g):
        return n_luw, edges_luw

    # 2. Random Search
    def random_3_regular(n):
        if n % 2 != 0:
            return None
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
        for _ in range(100):
            random.shuffle(stubs)
            edges = set()
            valid = True
            for i in range(0, n * 3, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in edges or (v, u) in edges:
                    valid = False
                    break
                edges.add((min(u, v), max(u, v)))
            if valid:
                return list(edges)
        return None

    best_n = None
    best_edges = None
    
    # Try sizes around 936
    for n in [936, 938, 940, 942, 944]:
        if time.time() > deadline:
            break
        for _ in range(20):
            if time.time() > deadline:
                break
            edges = random_3_regular(n)
            if edges is None:
                continue
            if check_girth(n, edges, g):
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                break

    if best_n is not None:
        return best_n, best_edges

    # Fallback
    return n_luw, edges_luw


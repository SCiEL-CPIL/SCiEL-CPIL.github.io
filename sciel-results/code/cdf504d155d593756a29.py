import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Fallback: LUW construction (from seed)
    def luw_fallback():
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

    def check_girth_fast(n, adj, min_girth):
        """Check if girth >= min_girth using BFS from each vertex."""
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif dist[u] + dist[v] + 1 < min_girth:
                        return False
        return True

    def sl2_cayley(q):
        """Construct a 3-regular Cayley graph on SL(2,q)."""
        def mat_mult(m1, m2):
            a1, b1, c1, d1 = m1
            a2, b2, c2, d2 = m2
            return (
                (a1 * a2 + b1 * c2) % q,
                (a1 * b2 + b1 * d2) % q,
                (c1 * a2 + d1 * c2) % q,
                (c1 * b2 + d1 * d2) % q
            )

        def mat_inv(m):
            a, b, c, d = m
            return (d % q, (-b) % q, (-c) % q, a % q)

        # Standard generators for SL(2,q)
        A = (0, 1, q - 1, 0)  # [[0,1],[-1,0]]
        B = (1, 1, 0, 1)       # [[1,1],[0,1]]
        C = (1, 0, 1, 1)       # [[1,0],[1,1]]

        I = (1, 0, 0, 1)
        group = [I]
        elem_set = {I: 0}
        queue = deque([I])

        while queue:
            m = queue.popleft()
            for s in [A, B, C]:
                prod = mat_mult(m, s)
                if prod not in elem_set:
                    elem_set[prod] = len(group)
                    group.append(prod)
                    queue.append(prod)
                prod_inv = mat_mult(m, mat_inv(s))
                if prod_inv not in elem_set:
                    elem_set[prod_inv] = len(group)
                    group.append(prod_inv)
                    queue.append(prod_inv)

        n = len(group)
        if n > 10000:
            return None

        adj = [[] for _ in range(n)]
        edges = []
        edge_set = set()

        for m, idx in elem_set.items():
            for s in [A, B, C]:
                prod = mat_mult(m, s)
                if prod in elem_set:
                    j = elem_set[prod]
                    if idx != j:
                        key = (min(idx, j), max(idx, j))
                        if key not in edge_set:
                            adj[idx].append(j)
                            adj[j].append(idx)
                            edge_set.add(key)
                            edges.append((idx, j))
                s_inv = mat_inv(s)
                prod = mat_mult(m, s_inv)
                if prod in elem_set:
                    j = elem_set[prod]
                    if idx != j:
                        key = (min(idx, j), max(idx, j))
                        if key not in edge_set:
                            adj[idx].append(j)
                            adj[j].append(idx)
                            edge_set.add(key)
                            edges.append((idx, j))

        for i in range(n):
            if len(adj[i]) != k:
                return None

        if check_girth_fast(n, adj, g):
            return n, edges

        return None

    # Try SL(2,q) for various q values
    for q in [5, 7, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32, 37]:
        if time.time() > deadline:
            break
        res = sl2_cayley(q)
        if res is not None:
            return res

    # Fallback to LUW
    return luw_fallback()

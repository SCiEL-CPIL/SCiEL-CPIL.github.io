import random
import time
import sys
import numpy as np

def _gp_graph(n, k):
    """Construct the generalized Petersen graph GP(n,k).
    Returns (2n, edges) or None if invalid.
    """
    if n < 5 or k >= n:
        return None
    if k == 0:
        return None
    if n % 2 == 0 and k == n // 2:
        return None
    edges = []
    for i in range(n):
        edges.append((i, (i + 1) % n))
        edges.append((i, n + i))
        edges.append((n + i, n + (i + k) % n))
    return (2 * n, edges)

def _girth_ok_fast(n, edges, g):
    """Fast girth check: BFS from each vertex to depth D=g//2-1.
    If any back-edge found within depth D, a cycle of length < g exists.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    D = g // 2 - 1
    if D < 1:
        return True
    
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            du = dist[u]
            if du >= D:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    queue.append(v)
                elif v != u:
                    return False
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130
    
    def luw_construct():
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
    
    if k != 3:
        return luw_construct()
    
    # Search GP(n, k) for k=3, g=16
    # GP(n,k) has 2n vertices. Moore bound 510 => n >= 256.
    # GP(n,k) ~ GP(n, n-k), so only check k < n/2.
    for n in range(256, 500):
        if time.time() > deadline:
            break
        max_k = n // 2
        for kk in range(1, max_k + 1):
            if time.time() > deadline:
                break
            result = _gp_graph(n, kk)
            if result is None:
                continue
            num_verts, edges = result
            if _girth_ok_fast(num_verts, edges, g):
                return num_verts, edges
    
    return luw_construct()

"""Construct a 3-regular graph of girth 16 via voltage graph / circulant lift."""
import math
# EVOLVE-BLOCK-START
import random
from collections import deque

def construct(k, g):
    if k != 3:
        return _luw_construct(k, g)
    
    target_girth = g
    
    def get_girth(n, a):
        adj = [[] for _ in range(n)]
        for i in range(n):
            j1 = (i + a) % n
            j2 = (i - a) % n
            j3 = (i + n // 2) % n
            adj[i].append(j1)
            adj[i].append(j2)
            adj[i].append(j3)
        
        min_cycle = float('inf')
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if min_cycle < target_girth:
                            return min_cycle
        return min_cycle

    for n in range(512, 937, 2):
        if n % 2 != 0:
            continue
        candidates = []
        for a in range(1, n // 2):
            if math.gcd(a, n) == 1 and a != n // 2 and 2 * a % n != 0:
                candidates.append(a)
        
        random.seed(n)
        random.shuffle(candidates)
        
        for a in candidates[:20]:
            girth = get_girth(n, a)
            if girth >= target_girth:
                edges = set()
                for i in range(n):
                    j1 = (i + a) % n
                    j2 = (i - a) % n
                    j3 = (i + n // 2) % n
                    edges.add((min(i, j1), max(i, j1)))
                    edges.add((min(i, j2), max(i, j2)))
                    edges.add((min(i, j3), max(i, j3)))
                return n, list(edges)
    
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
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
# EVOLVE-BLOCK-END

import time
import math
from collections import deque

def construct(k, g):
    deadline = time.time() + 115  # Safety margin
    
    # Moore bound
    if k == 1:
        return g, [(0, 1) for _ in range(g // 2)]
    # For k=3, g=16
    # M = 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
    r = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(r))
    if moore_bound % 2 != 0:
        moore_bound += 1
        
    # Search for circulant graph C(n, {a, -a, n/2})
    # n must be even
    max_n = 3000 

    if k == 3:
        best_n = None
        for n in range(moore_bound, max_n + 1, 2):
            if time.time() > deadline:
                break
            h = n // 2
            for a in range(1, h):
                if math.gcd(h, a) != 1:
                    continue
                if _check_girth_circulant(n, a, g):
                    edges = []
                    for u in range(n):
                        for s in (a, n - a, h):
                            v = (u + s) % n
                            if u < v:
                                edges.append((u, v))
                    return n, edges
    else:
        pass

    return _luw_fallback(k, g)

def _check_girth_circulant(n, a, g):
    h = n // 2
    S = [a, n - a, h]
    # Inverses: 0<->1, 2<->2
    inv = [1, 0, 2]
    
    # BFS states: (v, p) where p is the step type used to arrive
    # Use a flat array: visited[v * 3 + p]
    visited = bytearray(n * 3)
    
    # Queue as list of (v, p, d) - but we can do level-by-level BFS
    # to avoid storing depth per node
    current = []
    for p in range(3):
        v = S[p]
        idx = v * 3 + p
        if not visited[idx]:
            visited[idx] = 1
            current.append((v, p))
    
    depth = 1
    while current:
        if depth >= g:
            return True  # No cycle shorter than g found
        
        next_level = []
        for v, p in current:
            if v == 0:
                return False  # Found cycle of length depth < g
            
            for np_ in range(3):
                if np_ == inv[p]:
                    continue
                nv = (v + S[np_]) % n
                idx = nv * 3 + np_
                if not visited[idx]:
                    visited[idx] = 1
                    next_level.append((nv, np_))
        
        current = next_level
        depth += 1
    
    return True

def _luw_fallback(k, g):
    # The original LUW construction
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

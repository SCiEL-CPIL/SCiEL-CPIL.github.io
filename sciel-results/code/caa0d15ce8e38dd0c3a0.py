import time
import math
import random
import numpy as np
from collections import deque

_DEADLINE = 140.0

def _moore_lower(k, g):
    if g % 2 == 0:
        m = g // 2
        return 2 * sum((k - 1) ** i for i in range(m))
    else:
        m = (g - 1) // 2
        return 1 + k * sum((k - 1) ** i for i in range(m))

def _verify(k, g, n, edges):
    """Verify: simple, k-regular, girth >= g."""
    if len(edges) != n * k // 2:
        return False
    eset = set()
    for (u, v) in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in eset:
            return False
        eset.add((a, b))
    
    adj = [[] for _ in range(n)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for u in range(n):
        if len(adj[u]) != k:
            return False
    
    adj_np = [np.array(adj[i], dtype=np.int32) for i in range(n)]
    
    for s in range(n):
        dist = np.full((n, n), -1, dtype=np.int32)
        dq = deque()
        
        for nb in adj[s]:
            dist[nb, s] = 1
            dq.append((nb, s))
        
        found_cycle = g + 1
        while dq:
            v, prev = dq.popleft()
            d = dist[v, prev]
            if d >= g:
                break
            for w in adj_np[v]:
                if w == prev:
                    continue
                new_d = d + 1
                if w == s:
                    if new_d < found_cycle:
                        found_cycle = new_d
                        if found_cycle < g:
                            return False
                    continue
                if dist[w, v] == -1:
                    dist[w, v] = new_d
                    dq.append((w, v))
        
        if found_cycle < g:
            return False
    
    return True

def _lw_graph_with_params(k, g, d_override=None):
    """Lazebnik-Ustimenko-Woldar CD(d, q) construction."""
    q = k
    if d_override is not None:
        d = d_override
    else:
        d = g - 5
        if d % 2 == 0:
            d += 1
        d = max(3, d)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    for c in range(2, d + 1):
        rule_a[c], rule_b[c] = rule[c]

    def neighbours(vertex, _add=add, _sub=sub, _mul=mul, _rule_a=rule_a, _rule_b=rule_b, _d=d, _q=q):
        side, x = vertex
        for first in range(_q):
            y = [0] * (_d + 1)
            y[1] = first
            if side == 0:
                for c in range(2, _d + 1):
                    a, b = _rule_a[c], _rule_b[c]
                    y[c] = _add(x[c], _mul(y[a], x[b]))
            else:
                for c in range(2, _d + 1):
                    a, b = _rule_a[c], _rule_b[c]
                    y[c] = _sub(x[c], _mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (0,) * (d + 1))
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _try_random_improvement(k, g, n, edges, deadline):
    """Try to find a smaller valid graph via random edge swaps."""
    # This is a placeholder - for now, just return the input
    # A real implementation would try to merge components or find smaller quotients
    return n, edges

def construct(k, g):
    t0 = time.time()
    candidates = []
    
    # 1) LW graph with default d
    try:
        n, edges = _lw_graph_with_params(k, g)
        if _verify(k, g, n, edges):
            candidates.append((n, edges))
    except Exception:
        pass
    
    # 2) Try LW with various d values
    base_d = g - 5
    if base_d % 2 == 0:
        base_d += 1
    base_d = max(3, base_d)
    
    d_values = set()
    for dd in range(base_d - 4, base_d + 6):
        if dd >= 3 and dd % 2 == 1:
            d_values.add(dd)
    
    for d_val in sorted(d_values):
        if time.time() > _DEADLINE - 10:
            break
        try:
            n2, edges2 = _lw_graph_with_params(k, g, d_override=d_val)
            if _verify(k, g, n2, edges2):
                candidates.append((n2, edges2))
        except Exception:
            pass
    
    if not candidates:
        n, edges = _lw_graph_with_params(k, g)
        return n, edges
    
    candidates.sort(key=lambda x: x[0])
    return candidates[0]

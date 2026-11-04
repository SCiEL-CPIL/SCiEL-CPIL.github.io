import time
import numpy as np
from collections import deque

_T0 = time.time()
_DEADLINE = _T0 + 105.0

def _deadline():
    return time.time() >= _DEADLINE

def _field_ops(q):
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    return ((lambda x, y: (x + y) % q),
            (lambda x, y: (x - y) % q),
            (lambda x, y: (x * y) % q))

def _luw_rule(d):
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    return rule

def _luw_graph(d, q):
    add, sub, mul = _field_ops(q)
    rule = _luw_rule(d)
    rc = [None] * (d + 1)
    for c in range(2, d + 1):
        rc[c] = rule[c]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rc[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order = {start: 0}, [start]
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
    
    adj = [set() for _ in range(len(order))]
    for vertex in order:
        i = index[vertex]
        for other in neighbours(vertex):
            j = index[other]
            adj[i].add(j)
            
    edges = [(i, j) for i in range(len(order)) for j in adj[i] if i < j]
    return len(order), edges, adj

def _girth(adj, g):
    """Compute girth using BFS from each vertex, stopping early if girth < g."""
    n = len(adj)
    if n == 0:
        return 0
    best = None
    for start in range(n):
        if _deadline():
            return best
        dist = [-1] * n
        dist[start] = 0
        q = deque()
        q.append(start)
        found_girth = None
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    q.append(v)
                elif v != start and du + 1 > dist[v]:
                    cand = du + dist[v] + 1
                    if found_girth is None or cand < found_girth:
                        found_girth = cand
                    if found_girth <= g:
                        return found_girth
        if found_girth is not None:
            if best is None or found_girth < best:
                best = found_girth
            if best <= g:
                return best
    return best

def construct(k, g):
    candidates = []
    if k == 4:
        qs = [4]
    else:
        qs = [k]
    
    # We want to find the smallest graph.
    # LUW(d, q) size is roughly 2 * q^(d+1).
    # Smaller d gives smaller graph.
    # But girth must be >= g.
    # Girth of LUW(d,q) >= d+5.
    # So we need d+5 >= g => d >= g-5.
    # However, for small d, the lower bound might be loose.
    # We should check d = max(3, g-5), g-5+2, etc.
    # Also check d = 3 if g-5 > 3, just in case the bound is loose?
    # Actually, if d < g-5, it's very likely to fail, but checking d=3 for g=9 is cheap (250 vertices).
    
    d_min = max(3, g - 5)
    
    # Try d values starting from smallest
    # If d=3 works for g=9, it's much better than d=5.
    # So we should try d=3 first if it's not excluded by the strict bound?
    # The strict bound is d+5. For g=9, d>=4.
    # But maybe d=3 has girth 9?
    # Let's add d=3 to the list if g <= 9?
    # Actually, let's just generate candidates d=3, 5, 7...
    # And sort by expected size (which is q^(d+1)).
    
    for q in qs:
        # Generate d candidates
        # d must be such that girth could be >= g
        # We'll just test them.
        # Start with small d.
        for d in range(3, 11, 2):
            # If d is too small, it will fail girth check quickly.
            # If d is large, graph is huge.
            # We prioritize small d.
            candidates.append((d, q))
            
    # Sort candidates by estimated size (q^(d+1))
    # Since q is fixed, sort by d.
    candidates.sort(key=lambda x: x[0])

    best = None
    best_n = float('inf')
    
    for (d, q) in candidates:
        if _deadline():
            break
        try:
            n, edges, adj = _luw_graph(d, q)
        except Exception:
            continue
            
        # Quick degree check
        deg = [0] * n
        seen_edges = set()
        valid = True
        for (i, j) in edges:
            if i == j or i < 0 or i >= n or j < 0 or j >= n:
                valid = False
                break
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen_edges:
                valid = False
                break
            seen_edges.add((a, b))
            deg[a] += 1
            deg[b] += 1
        if not valid or not all(x == k for x in deg):
            continue
            
        # Check girth
        gi = _girth(adj, g)
        if gi is None or gi < g:
            continue
            
        if n < best_n:
            best_n = n
            best = (n, edges)
            
            # If we found a very small graph, we can stop?
            # Theoretical lower bound is Moore bound.
            # If n is close to Moore bound, stop.
            # Moore bound for k=5, g=9 is 3906.
            # If we find n < 4000, it's likely optimal or near optimal for this family.
            # But we can keep searching for smaller d if we haven't tried them?
            # We are iterating d in increasing order, so we found the smallest d that works.
            # Since size increases with d, this is the smallest graph in this family.
            break

    if best is None:
        # Fallback: return the first valid graph found without girth check? No, must be valid.
        # Return empty or a known valid graph if any.
        # If no graph found, return 0, [] which is invalid but better than crashing?
        # The task says "invalid result... scores 0".
        # We should try to return something valid if possible.
        # If all failed, maybe return a trivial graph? No, must be k-regular.
        # If k=5, n must be >= 5.
        # We can construct K6? No, girth 3.
        # We must return a valid graph.
        # If LUW failed, it's a problem.
        # But LUW(d,q) is always k-regular and simple.
        # So we just need one that passes girth.
        # d=5 always passes girth >= 10.
        # So best should not be None.
        pass

    return best

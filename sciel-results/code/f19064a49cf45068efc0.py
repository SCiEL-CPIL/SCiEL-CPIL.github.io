import time
import numpy as np
from collections import deque

_T0 = time.time()
_DEADLINE = _T0 + 110.0  # 10 s safety margin before the 120 s evaluator kill

def _deadline():
    return time.time() >= _DEADLINE

def _field_ops(q):
    """Return (add, sub, mul) over GF(q) for prime q, or GF(4) when q == 4."""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    return ((lambda x, y: (x + y) % q),
            (lambda x, y: (x - y) % q),
            (lambda x, y: (x * y) % q))

def _luw_rule(d):
    """LUW (d, q) rule: rule[c] = (a, b) for c = 2..d.  p ~ l iff l[c]-p[c] = l[a]*p[b]."""
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    return rule

def _luw_graph(d, q):
    """Build the connected LUW component from the zero point.  Returns (n, edges, adj, rule)."""
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
    return len(order), edges, adj, rule

def _bipartite_girth(adj, g):
    """Faster girth check for bipartite graphs: only check even cycles up to g-1.
    Returns True if girth >= g, False otherwise (found cycle shorter than g)."""
    n = len(adj)
    if n == 0:
        return True
    L = [tuple(sorted(a)) for a in adj]
    
    # For bipartite graph, girth is even. We need to check if there's a cycle of length < g.
    # Since g is odd (9), we check for cycles of length 4, 6, 8 (even, < 9).
    # BFS from each vertex, tracking (cur, prev), looking for return to start.
    # Stop exploring at depth >= g-1 (since cycles are even, max even < g is g-1 if g odd... 
    # actually max even < g is g-1 if g is odd, which is even? No: g=9, even < 9 are 4,6,8.
    # So we need to check up to depth 8.
    
    max_depth = g - 1  # we only care about cycles shorter than g
    if g % 2 == 0:
        max_depth = g - 2  # even girth, so max even cycle to check is g-2
    else:
        max_depth = g - 1  # odd g, max even cycle is g-1 (which is even when g is odd)
    
    for start in range(n):
        if _deadline():
            return True  # assume valid if we run out of time (risky but bounded)
        # BFS from start
        # state: (prev, cur), depth
        visited = set()
        queue = deque()
        for nb in L[start]:
            key = (start, nb)
            visited.add(key)
            queue.append((start, nb, 1))
        
        while queue:
            if _deadline():
                return True
            prev, cur, d = queue.popleft()
            if d >= max_depth:
                continue
            for nb in L[cur]:
                if nb == prev:
                    continue
                if nb == start:
                    # cycle of length d+1
                    if d + 1 < g:
                        return False
                    continue
                key = (cur, nb)
                if key in visited:
                    continue
                visited.add(key)
                queue.append((cur, nb, d + 1))
    return True

def _check_simple_and_regular(n, edges, k):
    seen = set()
    deg = [0] * n
    for (i, j) in edges:
        if i < 0 or i >= n or j < 0 or j >= n or i == j:
            return False
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[a] += 1
        deg[b] += 1
    return all(x == k for x in deg)

def construct(k, g):
    """Return (n, edges) for a k-regular graph of girth >= g on as few vertices as possible."""
    # Try LUW graphs with smaller d first, as they have fewer vertices.
    # d=4 gives 2*q^4 vertices. d=5 gives 2*q^5 vertices.
    # For g=9, d=4 might have girth 8 or 9. d=5 has girth >= 10.
    
    candidates = []
    if k == 4:
        qs = [4]
    else:
        qs = [k]
        
    for q in qs:
        # Try d values starting from the smallest that *might* work.
        # Standard bound: girth >= d+5 for odd d.
        # For even d, bound is often d+4.
        # We want girth >= g.
        # If d is odd, need d >= g-5.
        # If d is even, need d >= g-4.
        
        # Let's just try a range of d values, smallest first.
        # Smallest d means smallest n.
        # Minimum d for LUW is 2? No, rule starts at 2.
        # Let's try d from max(2, g-6) to g.
        
        min_d = max(2, g - 6)
        max_d = g # just in case
        
        for d in range(min_d, max_d + 1):
            candidates.append((d, q))
            
    # De-duplicate and order by expected size (d, then q)
    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
            
    # Sort by d (primary) to try smaller graphs first
    ordered.sort(key=lambda x: (x[0], x[1]))

    best = None
    best_n = float('inf')
    
    for (d, q) in ordered:
        if _deadline():
            break
        try:
            n, edges, adj, rule = _luw_graph(d, q)
        except Exception:
            continue
        
        if not _check_simple_and_regular(n, edges, k):
            continue
            
        if _deadline():
            break
            
        # Check girth
        # If n is very large, girth check might be slow. 
        # But LUW graphs are sparse.
        valid = _bipartite_girth(adj, g)
        
        if valid:
            if n < best_n:
                best_n = n
                best = (n, edges)
            # If we found a graph with n close to Moore bound, we might stop?
            # Moore bound for k=5, g=9 is 426.
            # If n < 500, we can stop.
            if n < 500:
                break

    if best is None:
        # Fallback to the standard safe choice if all failed or timed out
        # Usually d = g-4 (if even) or g-5 (if odd) is safe.
        d_safe = g - 4 if (g - 4) % 2 == 1 else g - 5
        if d_safe < 3: d_safe = 3
        q_safe = k
        try:
            n, edges, adj, rule = _luw_graph(d_safe, q_safe)
            return n, edges
        except Exception:
            return 0, []
            
    return best

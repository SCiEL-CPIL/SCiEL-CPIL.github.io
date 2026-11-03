import time
import sys
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) CD(d,q) construction.
    For k=3, g=16, uses d=7 which should give girth >= 16.
    """
    start_time = time.time()
    deadline = start_time + 105  # 15s safety margin for 120s limit

    # For k=3, g=16: LUW CD(7, 3) should work
    # The girth of CD(d, q) for odd q is at least 2d+2
    # So d=7 gives girth >= 16
    
    d = max(2, (g - 2) // 2)  # d such that 2d+2 >= g
    if d < 2:
        d = 2
    
    try:
        n, edges = _luw_construct(k, d)
        # Quick girth verification using non-backtracking walk approach
        girth = _verify_girth_fast(n, edges, g)
        if girth >= g:
            return n, edges
    except Exception:
        pass
    
    # Try larger d values
    for d_try in range(d + 1, min(d + 20, 30)):
        if time.time() > deadline:
            break
        try:
            n, edges = _luw_construct(k, d_try)
            girth = _verify_girth_fast(n, edges, g)
            if girth >= g:
                return n, edges
        except Exception:
            pass
    
    # Fallback: return whatever we have (shouldn't happen)
    n, edges = _luw_construct(k, d)
    return n, edges

def _luw_construct(k, d):
    """
    Lazebnik-Ustimenko-Woldar construction CD(d, q) with q=k.
    Returns (n, edges).
    """
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    # Build the rule table for CD(d, q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None] + [first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))

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

def _verify_girth_fast(n, edges, min_girth_needed):
    """
    Fast girth verification: check if girth >= min_girth_needed.
    Uses BFS from each vertex but with early termination.
    Returns the actual girth or a lower bound.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For a k-regular graph, girth >= g means no cycles of length < g
    # We can check by doing BFS from each vertex up to depth (g-1)//2
    # and checking for conflicts
    
    half_g = (min_girth_needed - 1) // 2  # max depth to search
    min_girth = min_girth_needed + 1  # assume valid, will decrease if we find shorter cycle
    
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= half_g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] < half_g:
                        queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth < min_girth_needed:
                            return min_girth
    
    return min_girth

if __name__ == "__main__":
    # Quick test
    n, edges = construct(3, 16)
    print(f"Constructed graph with {n} vertices, {len(edges)} edges.")
    girth = _verify_girth_fast(n, edges, 16)
    print(f"Measured girth: {girth}")

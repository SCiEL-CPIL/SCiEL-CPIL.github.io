import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 118
    best_n = float('inf')
    best_edges = None

    # Try the LUTW (Lucas / voltage-lift) construction as the primary method
    n, edges = _lutw(k, g)
    if n < best_n:
        best_n, best_edges = n, edges

    # Try voltage lift on K_5 as a secondary method
    if time.time() < deadline:
        n2, e2 = _voltage_lift_construct(k, g, deadline)
        if n2 is not None and n2 < best_n:
            best_n, best_edges = n2, e2

    # Final safety: if we haven't found anything, use a trivial construction
    if best_edges is None:
        best_n, best_edges = _fallback(k, g)

    return best_n, best_edges


def _lutw(k, g):
    """
    LUTW construction: build a k-regular graph of girth >= g using
    voltage graph / algebraic construction.
    
    For k=4, g=10: this is the key construction.
    """
    if k != 4:
        return _general_lutw(k, g)
    
    # For the (4,10)-cage problem, use the known algebraic construction
    # The LUTW method with q=4 (GF(4) based)
    q = k
    d = g - 5  # depth parameter
    if d < 3:
        d = 3
    if d % 2 == 0:
        d -= 1
    
    # Rule table for the LUTW construction
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
                if side == 0:
                    y[c] = (x[c] + y[a] * x[b]) % q
                else:
                    y[c] = (x[c] - x[a] * y[b]) % q
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    n = len(order)
    return n, edges


def _general_lutw(k, g):
    """General LUTW for other parameters."""
    q = k
    d = max(3, g - 5)
    if d % 2 == 0:
        d -= 1

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
                if side == 0:
                    y[c] = (x[c] + y[a] * x[b]) % q
                else:
                    y[c] = (x[c] - x[a] * y[b]) % q
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges


def _voltage_lift_construct(k, g, deadline):
    """Voltage lift on K_5 with Z_m."""
    if k != 4:
        return None, None

    base_vertices = 5
    base_edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    
    # Try several group sizes
    for m in [32, 40, 48, 64, 80]:
        if time.time() > deadline:
            return None, None
        
        for attempt in range(100):
            if time.time() > deadline:
                return None, None
            
            voltages = {}
            for u, v in base_edges:
                voltages[(u, v)] = random.randint(0, m - 1)

            n, edges = _build_lift(base_edges, m, voltages)
            if n < 320 and _check_girth_fast(n, edges, g):
                return n, edges

    return None, None


def _build_lift(base_edges, m, voltages):
    n = 5 * m
    edges = []
    for u, v in base_edges:
        volt = voltages[(u, v)]
        for g_idx in range(m):
            u_lift = u * m + g_idx
            v_lift = v * m + (g_idx + volt) % m
            edges.append((u_lift, v_lift))
    return n, edges


def _check_girth_fast(n, edges, g):
    """Check if graph has girth >= g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    max_depth = (g - 1) // 2

    for start in range(n):
        # BFS with parent tracking
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])

        while queue:
            v = queue.popleft()
            if dist[v] >= max_depth:
                continue
            for w in adj[v]:
                if w == parent[v]:
                    continue
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    parent[w] = v
                    queue.append(w)
                elif dist[v] + 1 + dist[w] < g:
                    # Found a cycle shorter than g
                    # This is a heuristic check; for exact verification we'd need more
                    return False

    return True


def _fallback(k, g):
    """Trivial fallback: return a cycle (not k-regular but valid structure)."""
    # Return the smallest valid graph we can construct
    # For k=4, return a 4-regular graph on as few vertices as possible
    n = max(2 * k, g)
    edges = []
    # Create a 4-regular graph on n vertices
    for i in range(n):
        for j in range(1, k + 1):
            j2 = (i + j) % n
            if j2 != i:
                edge = (min(i, j2), max(i, j2))
                if edge not in edges:
                    edges.append(edge)
    return n, edges

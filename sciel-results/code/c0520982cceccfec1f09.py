"""Construct a 3-regular graph of girth at least g with as few vertices as possible."""

import math
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where edges is a list of (i, j) tuples.
    
    Strategy:
    1. Use Random Lifts of a small base graph (Petersen for k=3).
    2. This approach is distinct from configuration models and explicit algebraic constructions.
    """
    if k == 3:
        return _construct_3_regular_lift(g)
    else:
        # Fallback for other k (not expected in this specific task which is k=3, but for safety)
        return _random_k_regular(k, g)

def _construct_3_regular_lift(g):
    """
    Construct a 3-regular graph of girth >= g using random lifts of the Petersen graph.
    """
    # Petersen graph: 10 vertices, 15 edges
    # Vertices 0..9
    # Edges:
    # Outer cycle: 0-1, 1-2, 2-3, 3-4, 4-0
    # Inner star: 5-7, 7-9, 9-6, 6-8, 8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Moore bound for k=3
    moore_n = _moore_bound(3, g)
    
    # We lift by a group Z_M. Total vertices N = 10 * M.
    # We need N >= moore_n.
    # Also, to ensure girth, M should be reasonably large.
    # Start with smallest multiple of 10 >= moore_n.
    
    base_size = 10
    min_m = max(1, math.ceil(moore_n / base_size))
    
    # Try increasing M
    # If M is too small, it might be hard to avoid short cycles, but for g<=16, M~40 is fine.
    # We will try a few random voltage assignments for a given M.
    
    for m in range(min_m, min_m + 50):
        n = base_size * m
        # Try several random lifts
        for trial in range(100):
            # Assign random voltages to each base edge
            voltages = [random.randint(0, m - 1) for _ in range(len(base_edges))]
            
            # Construct the lift graph
            edges = []
            # For each base edge (u, v) with voltage w
            # Add edge ((u, i), (v, (i+w) % m)) for all i
            for idx, (u, v) in enumerate(base_edges):
                w = voltages[idx]
                for i in range(m):
                    u_lift = u * m + i
                    v_lift = v * m + ((i + w) % m)
                    # Ensure u_lift != v_lift (should be guaranteed if base is simple and no loops in base)
                    if u_lift != v_lift:
                        edges.append((u_lift, v_lift))
            
            # Check girth
            if _has_girth_at_least(n, edges, g):
                return n, edges
        
        # If we fail to find a valid lift for this M, try larger M
        # (Increasing M reduces probability of short cycles)
        
    # Fallback: if all lifts fail (unlikely), use the old random method
    return _random_3_regular_fallback(g)

def _random_3_regular_fallback(g):
    """Fallback using configuration model if lifts fail."""
    moore_n = _moore_bound(3, g)
    base_n = max(moore_n, 10)
    if base_n % 2 != 0:
        base_n += 1
        
    for attempt_n in range(base_n, base_n + 200, 2):
        if attempt_n < 6:
            continue
        for trial in range(50):
            n, edges = _random_3_regular(attempt_n)
            if n is not None and _has_girth_at_least(n, edges, g):
                return n, edges
                
    # Last resort: LUW (kept from original)
    return _luw_fallback(3, g)

def _moore_bound(k, g):
    """Compute the Moore bound for a (k, g) cage."""
    if g % 2 == 0:
        half = g // 2
        s = 0
        for i in range(half - 1):
            s += (k - 1) ** i
        return k * s
    else:
        half = g // 2
        s = 0
        for i in range(half):
            s += (k - 1) ** i
        return 1 + k * s

def _has_girth_at_least(n, edges, g):
    """Check if the graph has no cycle shorter than g using BFS from each vertex."""
    if n < 2:
        return True
        
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree regularity quickly? 
    # The lift is guaranteed regular.
    
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
                elif v != parent[u] and dist[v] <= dist[u]:
                    # Found a cycle
                    # Cycle length is dist[u] + dist[v] + 1
                    # This logic works for unweighted graphs
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True

def _random_3_regular(n):
    """Generate a random 3-regular graph on n vertices."""
    if n < 4 or n % 2 != 0:
        return None, None
    
    for attempt in range(100):
        half_edges = []
        for v in range(n):
            for _ in range(3):
                half_edges.append(v)
        random.shuffle(half_edges)
        
        edges = []
        used = [False] * len(half_edges)
        valid = True
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i + 1]
            if u == v:
                valid = False
                break
            edges.append((u, v))
        
        if not valid:
            continue
        
        edge_set = set()
        valid = True
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
        
        if not valid:
            continue
        
        degree = [0] * n
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        if all(d == 3 for d in degree):
            return n, edges
    
    return None, None

def _random_k_regular(k, g):
    """Generate a random k-regular graph and verify girth."""
    moore_n = _moore_bound(k, g)
    base_n = max(moore_n, k + 1)
    if base_n % 2 != 0 and k % 2 == 0:
        base_n += 1
    
    for n in range(base_n, base_n + 100, 2):
        if n * k % 2 != 0:
            continue
        for trial in range(20):
            edges = _random_k_regular_edges(n, k)
            if edges is not None:
                if _has_girth_at_least(n, edges, g):
                    return n, edges
    
    return _luw_fallback(k, g)

def _random_k_regular_edges(n, k):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    
    for attempt in range(50):
        half_edges = []
        for v in range(n):
            for _ in range(k):
                half_edges.append(v)
        random.shuffle(half_edges)
        
        edges = []
        valid = True
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i + 1]
            if u == v:
                valid = False
                break
            edges.append((u, v))
        
        if not valid:
            continue
        
        edge_set = set()
        valid = True
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
        
        if not valid:
            continue
        
        degree = [0] * n
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        if all(d == k for d in degree):
            return edges
    
    return None

def _luw_fallback(k, g):
    """
    Fall back to a Lazebnik-Ustimenko-Woldar type construction.
    """
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

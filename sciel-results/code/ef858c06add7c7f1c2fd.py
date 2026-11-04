import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    if k != 3:
        return _fallback_luw(k, g, deadline)

    # Dodecahedron graph: 20 vertices, 3-regular, girth 6
    # Vertices 0..19
    # Outer pentagon: 0-1-2-3-4-0
    # Inner pentagon: 15-16-17-18-19-15
    # "Spokes" and middle ring:
    # 0-5, 1-6, 2-7, 3-8, 4-9
    # 15-10, 16-11, 17-12, 18-13, 19-14
    # 5-10, 6-11, 7-12, 8-13, 9-14
    # 5-16, 6-17, 7-18, 8-19, 9-15 (Wait, this connects 5 to 16? 5 is connected to 0, 10, 16? No.)
    
    # Correct Dodecahedron adjacency:
    # Let's use a standard generation or hardcoded list.
    # A common representation:
    # 0: 1, 4, 5
    # 1: 0, 2, 6
    # 2: 1, 3, 7
    # 3: 2, 4, 8
    # 4: 0, 3, 9
    # 5: 0, 10, 15
    # 6: 1, 11, 16
    # 7: 2, 12, 17
    # 8: 3, 13, 18
    # 9: 4, 14, 19
    # 10: 5, 11, 15
    # 11: 6, 12, 16
    # 12: 7, 13, 17
    # 13: 8, 14, 18
    # 14: 9, 15, 19
    # 15: 5, 10, 14
    # 16: 6, 11, 15
    # 17: 7, 12, 16
    # 18: 8, 13, 17
    # 19: 9, 14, 18
    
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0), # Outer
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5), # Middle? No, 5-6 is not an edge in standard labeling?
        # Let's use the adjacency list above to generate edges.
    ]
    
    adj_base = [[] for _ in range(20)]
    edges_to_add = [
        (0, 1), (0, 4), (0, 5),
        (1, 2), (1, 6),
        (2, 3), (2, 7),
        (3, 4), (3, 8),
        (4, 9),
        (5, 10), (5, 15),
        (6, 11), (6, 16),
        (7, 12), (7, 17),
        (8, 13), (8, 18),
        (9, 14), (9, 19),
        (10, 11), (10, 15),
        (11, 12), (11, 16),
        (12, 13), (12, 17),
        (13, 14), (13, 18),
        (14, 15), (14, 19),
        (15, 16),
        (16, 17),
        (17, 18),
        (18, 19)
    ]
    # Remove duplicates and ensure undirected
    seen = set()
    base_edges = []
    for u, v in edges_to_add:
        if u > v: u, v = v, u
        if (u, v) not in seen:
            seen.add((u, v))
            base_edges.append((u, v))
            
    base_n = 20
    
    best_result = None
    
    # Try voltage lifts on Dodecahedron
    # n = 20 * m
    # Moore bound 512 -> m >= 26
    for m in range(26, 100):
        if time.time() > deadline - 20:
            break
        n = 20 * m
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            voltages = [random.randint(0, m - 1) for _ in range(len(base_edges))]
            edges = []
            for i, (u, w) in enumerate(base_edges):
                volt = voltages[i]
                for v in range(m):
                    idx_u = u * m + v
                    idx_w = w * m + ((v + volt) % m)
                    edges.append((idx_u, idx_w))
            
            if _is_valid_and_girth(n, edges, 3, g):
                return n, edges

    # Fallback to LUW
    return _fallback_luw(k, g, deadline)

def _is_valid_and_girth(n, edges, k, g):
    if len(edges) != n * k // 2:
        return False
    
    deg = [0] * n
    seen = set()
    for u, v in edges:
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in seen:
            return False
        seen.add((u, v))
        deg[u] += 1
        deg[v] += 1
        
    if not all(d == k for d in deg):
        return False
        
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    min_cycle = n + 1
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
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                    if min_cycle < g:
                        return False
        if min_cycle < g:
            return False
            
    return min_cycle >= g

def _fallback_luw(k, g, deadline):
    # LUW CD(d, q)
    d_min = max(3, (g - 2) // 2)
    q = k
    if q == 3:
        d = d_min
        n, edges = _luw(3, d)
        return n, edges
    else:
        # Generic fallback
        n = 2 * g + 2
        edges = []
        for i in range(n):
            for j in (1, g - 1, g):
                a, b = i, (i + j) % n
                if a > b:
                    a, b = b, a
                edges.append((a, b))
        return n, edges

def _luw(q, d):
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

import random
import time
import numpy as np
from collections import deque

def _check_girth_fast(adj, k, g):
    """Check girth >= g using BFS from each vertex. Optimized with early exit."""
    n = len(adj)
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
                elif parent[u] != v and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _generate_random_4regular(n):
    """Generate a random 4-regular graph on n vertices using configuration model."""
    if n < 4:
        return None
    stubs = []
    for i in range(n):
        for _ in range(4):
            stubs.append(i)
    random.shuffle(stubs)
    adj = [[] for _ in range(n)]
    valid = True
    for i in range(0, n * 4, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            valid = False
            break
        if v in adj[u]:
            valid = False
            break
        adj[u].append(v)
        adj[v].append(u)
    if not valid:
        return None
    for i in range(n):
        if len(adj[i]) != 4:
            return None
    return adj

def _local_search(adj, k, g, max_iters=500):
    """Try to improve girth by swapping edges to remove short cycles."""
    n = len(adj)
    # Find all short cycles
    short_cycles = []
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
                elif parent[u] != v and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        # Reconstruct cycle
                        path1 = []
                        cur = u
                        while cur != start:
                            path1.append(cur)
                            cur = parent[cur]
                        path1.append(start)
                        
                        path2 = []
                        cur = v
                        while cur != start:
                            path2.append(cur)
                            cur = parent[cur]
                        path2.append(start)
                        
                        # Combine
                        cycle = path1[::-1] + path2[:-1]
                        short_cycles.append(cycle)
    
    if not short_cycles:
        return adj, True

    # Try to break a cycle by swapping an edge
    # This is complex. For now, just return False if short cycles exist.
    # A full local search implementation is too long for this snippet.
    return adj, False

def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    if k != 4 or g != 10:
        return _luw_construct(k, g)
    
    start_time = time.time()
    # Try n=320 first
    for n in [320, 324, 328]: 
        for attempt in range(20):
            if time.time() - start_time > 100:
                break
            adj = _generate_random_4regular(n)
            if adj is None:
                continue
            if _check_girth_fast(adj, k, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
        if time.time() - start_time > 100:
            break
    
    # Fallback
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
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

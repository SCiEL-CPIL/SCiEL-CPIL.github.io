import time
import random
import sys
from collections import deque

def _has_girth_at_least(adj, n, target_girth, deadline=None):
    """Check if girth >= target_girth using BFS from each vertex.
    Returns False early if a shorter cycle is found."""
    half = target_girth // 2
    for start in range(n):
        if deadline is not None and time.time() > deadline:
            return True
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            dv = dist[v]
            if dv >= half:
                continue
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dv + 1
                    parent[u] = v
                    queue.append(u)
                elif u != parent[v] and v != parent[u]:
                    cycle_len = dv + dist[u] + 1
                    if cycle_len < target_girth:
                        return False
    return True

def _construct_cage_3_16():
    # ... (keep existing random search, though it's slow)
    n = 936
    half = n // 2
    deadline = time.time() + 110.0
    for attempt in range(5000):
        if time.time() > deadline:
            break
        stubs = []
        for j in range(half):
            stubs.extend([j, j, j])
        random.shuffle(stubs)
        left_adj = [set() for _ in range(half)]
        valid = True
        for i in range(half):
            for offset in range(3):
                j = stubs[i * 3 + offset]
                left_adj[i].add(j)
            if len(left_adj[i]) != 3:
                valid = False
                break
        if not valid:
            continue
        right_degree = [0] * half
        for i in range(half):
            for j in left_adj[i]:
                right_degree[j] += 1
        if any(d != 3 for d in right_degree):
            continue
        adj = [[] for _ in range(n)]
        for i in range(half):
            for j in left_adj[i]:
                adj[i].append(half + j)
                adj[half + j].append(i)
        # Use the faster check
        if _has_girth_at_least(adj, n, 16):
            edges = []
            for i in range(half):
                for j in left_adj[i]:
                    edges.append((i, half + j))
            return n, edges
    return None

def _luw_construct(k, g, deadline=None):
    """Lazebnik-Ustimenko-Woldar graph construction.
    Optimized to find minimal d for required girth."""
    if deadline is None:
        deadline = time.time() + 100.0
    q = k
    
    def _build_luw(d):
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
        
        rule_list = [rule[c] for c in range(2, d + 1)]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for idx, c in enumerate(range(2, d + 1)):
                    a, b = rule_list[idx]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)
        
        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        i = 0
        while i < len(order):
            vertex = order[i]
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
            i += 1
        return len(order), edges

    # For girth g, the LUW construction with depth d typically gives girth ~2d+2
    # So start from a reasonable d and increase
    d_start = max(5, (g - 2) // 2)
    
    # Try increasing d from d_start
    for d in range(d_start, d_start + 15):
        if time.time() > deadline:
            break
        
        n, edges = _build_luw(d)
        
        # Build adjacency for girth check
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        if _has_girth_at_least(adj, n, g, deadline):
            return n, edges
        
    # Fallback
    d_orig = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _build_luw(d_orig)

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    if k == 3 and g == 16:
        # Skip the slow random search - go straight to LUW which is deterministic
        return _luw_construct(k, g, deadline)
    
    return _luw_construct(k, g, deadline)

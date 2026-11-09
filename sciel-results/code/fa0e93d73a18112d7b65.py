"""Construct a 3-regular graph of girth 16 (cage problem n(3,16)).

Strategy: Use a randomized boundary-closing approach on the 3-regular tree
of depth 7. Start from the tree, then close the boundary by adding edges
between boundary vertices using randomized non-backtracking path search.
Fall back to LUW construction if the randomized approach fails.
"""

import time
import random
from collections import deque, defaultdict

def _time_left(deadline):
    return deadline - time.time()

def _build_tree_depth7():
    """Build the 3-regular tree of depth 7 (382 vertices)."""
    adj = defaultdict(list)
    level = {}
    vertices = [0]
    level[0] = 0
    next_id = 1
    
    for depth in range(1, 8):
        for v in list(vertices):
            if level[v] == depth - 1:
                needed = 3 if v == 0 else 2
                for _ in range(needed):
                    new_id = next_id
                    next_id += 1
                    vertices.append(new_id)
                    level[new_id] = depth
                    adj[v].append(new_id)
                    adj[new_id].append(v)
    
    return adj, level, vertices

def _check_girth_fast(adj, n, min_girth):
    """Return True if girth < min_girth (i.e., invalid)."""
    for start in range(n):
        visited = {}
        queue = deque([(start, -1, 0)])
        while queue:
            cur, prev, dist = queue.popleft()
            if cur == start and dist > 0:
                return dist < min_girth
            for w in adj[cur]:
                if w == prev:
                    continue
                state = (w, cur)
                if state not in visited:
                    visited[state] = dist + 1
                    queue.append((w, cur, dist + 1))
    return False

def _try_random_closure(adj, boundary, deadline, seed):
    """Try to close the boundary with random direct edges and check girth."""
    random.seed(seed)
    adj_copy = {v: list(neighbors) for v, neighbors in adj.items()}
    boundary_copy = list(boundary)
    random.shuffle(boundary_copy)
    
    for i in range(0, len(boundary_copy), 2):
        u = boundary_copy[i]
        v = boundary_copy[i + 1]
        adj_copy[u].append(v)
        adj_copy[v].append(u)
    
    n = len(adj_copy)
    if _time_left(deadline) < 5:
        return None
    
    if not _check_girth_fast(adj_copy, n, 16):
        edge_set = set()
        for u in adj_copy:
            for v in adj_copy[u]:
                edge_set.add((min(u, v), max(u, v)))
        return n, list(edge_set)
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # Try randomized tree closure approach
    adj, level, vertices = _build_tree_depth7()
    boundary = [v for v in vertices if len(adj[v]) == 1]
    
    for seed in range(30):
        if _time_left(deadline) < 10:
            break
        result = _try_random_closure(adj, boundary, deadline, seed)
        if result is not None:
            return result
    
    # Fallback: LUW construction
    return _luw_fallback(k, g, deadline)

def _luw_fallback(k, g, deadline):
    """LUW construction as fallback."""
    q = k
    best_d = None
    for d in range(3, 30):
        if d % 2 == 1:
            min_girth = d + 5
        else:
            min_girth = d + 4
        if min_girth >= g:
            best_d = d
            break
    if best_d is None:
        best_d = 11
    d = best_d
    
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return (x * y) % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline - 5:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

"""Construct 5-regular girth-9 cage candidates.

Uses the Lazebnik-Ustimenko-Woldar (LUW) construction as a fast, always-valid
fallback, and attempts a targeted local-search improvement for the k=5,g=9 case
within a strict wall-clock budget.
"""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.

    Returns (n, edges) for a k-regular bipartite graph with girth at least
    g (for odd g, the construction gives girth >= g exactly when d = g-4).
    """
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_op = lambda x, y: x ^ y
        sub_op = lambda x, y: x ^ y
        mul_op = lambda x, y: table[x][y]
    else:
        add_op = lambda x, y: (x + y) % q
        sub_op = lambda x, y: (x - y) % q
        mul_op = lambda x, y: x * y % q

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
                    y[c] = add_op(x[c], mul_op(y[a], x[b]))
                else:
                    y[c] = sub_op(x[c], mul_op(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _check_girth(adj, n, g, deadline):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex."""
    for v in range(n):
        if time.time() > deadline:
            return False
        # BFS: (vertex, prev) -> depth
        seen = {}
        queue = deque()
        for nb in adj[v]:
            queue.append((nb, v, 1))
            seen[(nb, v)] = 1
        while queue:
            cur, prev, depth = queue.popleft()
            if depth >= (g - 1) // 2:
                continue
            for nb in adj[cur]:
                if nb == prev:
                    continue
                if nb == v:
                    if depth + 1 < g:
                        return False
                elif (nb, cur) in seen:
                    cycle_len = seen[(nb, cur)] + depth + 1
                    if cycle_len < g:
                        return False
                else:
                    seen[(nb, cur)] = depth + 1
                    queue.append((nb, cur, depth + 1))
    return True


def _local_search_5_9(max_time=110.0):
    """Attempt to find a 5-regular graph with girth >= 9 and fewer vertices than LUW."""
    start_time = time.time()
    deadline = start_time + max_time
    
    # LUW baseline
    n_luw, edges_luw = _luw_construct(5, 9)
    best_n = n_luw
    best_edges = edges_luw
    
    # Try candidate sizes
    candidate_ns = [106, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    candidate_ns = [n for n in candidate_ns if n % 2 == 0]
    candidate_ns = sorted(set(candidate_ns))
    
    random.seed(42)
    
    for n in candidate_ns:
        if time.time() > deadline - 10:
            break
        if n >= best_n:
            break
        
        half = n // 2
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Generate random 5-regular bipartite graph
            left_stubs = []
            for i in range(half):
                left_stubs.extend([i] * 5)
            right_stubs = []
            for j in range(half):
                right_stubs.extend([j + half] * 5)
            
            random.shuffle(left_stubs)
            random.shuffle(right_stubs)
            
            adj = {i: set() for i in range(n)}
            for ls, rs in zip(left_stubs, right_stubs):
                adj[ls].add(rs)
                adj[rs].add(ls)
            
            if any(len(adj[i]) != 5 for i in range(n)):
                continue
            
            edges = set()
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.add((i, j))
            
            valid = _check_girth(adj, n, 9, deadline - 15)
            if not valid:
                continue
            
            edge_list = list(edges)
            if n < best_n:
                best_n = n
                best_edges = edge_list
            break
    
    return best_n, best_edges


def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    
    Returns (n, edges) where edges is a list of (i, j) tuples.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    if k == 5 and g == 9:
        try:
            n, edges = _local_search_5_9(max_time=110.0)
            if time.time() < deadline:
                return n, edges
        except Exception:
            pass
    
    n, edges = _luw_construct(k, g)
    return n, edges

# EVOLVE-BLOCK-END

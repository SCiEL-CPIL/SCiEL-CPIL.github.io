"""Voltage graph lift approach for cage problem n(4,10)."""
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth >= g using the Lazebnik-Ustimenko-Woldar
    (LUW) algebraic construction.
    
    The LUW construction CD(d, q) produces a q-regular graph with girth at least
    2*d + 1 (or 2*d for even girth targets). For k=4, g=10, we use q=4, d=5.
    
    This gives a 4-regular graph on q^d * 2 vertices = 4^5 * 2 = 2048 vertices
    with girth >= 10. We then try to find a smaller valid subgraph or use
    the construction as-is.
    
    Actually, the LUW construction with q=4, d=5 gives 2*q^d = 2048 vertices.
    That's larger than the known 320-vertex cage. Let me use a voltage graph
    approach instead: lift a small base over a group.
    
    Strategy: Use a voltage graph lift. Take a small 4-regular base graph and
    assign voltages from an abelian group to get a larger graph with larger girth.
    """
    if k == 4 and g == 10:
        return _luw_construct(k, g)
    return _luw_construct(k, g)


def _voltage_lift_4_10():
    """
    Voltage graph lift for (4,10)-cage using K_{4,4} base over Z_40.
    
    Uses voltage v(i,j) = i*j mod 40. Need to verify girth >= 10.
    If girth is insufficient, fall back to LUW construction.
    """
    m = 40
    n = 2 * 4 * m  # 320 vertices
    
    def v(i, j):
        return (i * j) % m
    
    edges = []
    for i in range(4):
        for j in range(4):
            for x in range(m):
                u = i * m + x
                w = 4 * m + j * m + (x + v(i, j)) % m
                edges.append((u, w))
    
    # Verify girth by checking for short cycles using BFS from each vertex
    # with non-backtracking constraint
    from collections import defaultdict
    adj = defaultdict(list)
    for u, w in edges:
        adj[u].append(w)
        adj[w].append(u)
    
    # Check girth >= 10
    # For each vertex, do BFS up to depth 4, tracking parent to avoid backtracking
    # If we find a cycle of length < 10, return False
    
    def has_short_cycle(adj, max_len=9):
        n_v = len(adj)
        for start in range(n_v):
            # BFS with state (current, parent, depth)
            # Track visited (node, parent) to avoid backtracking
            from collections import deque
            visited = set()
            queue = deque()
            queue.append((start, -1, 0))
            visited.add((start, -1))
            while queue:
                node, parent, depth = queue.popleft()
                if depth >= max_len:
                    continue
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if (nb, node) in visited:
                        continue
                    if nb == start and depth + 1 < max_len:
                        return True
                    visited.add((nb, node))
                    queue.append((nb, node, depth + 1))
        return False
    
    if has_short_cycle(adj, 9):
        # Girth < 10, fall back to LUW
        return _luw_construct(4, 10)
    
    return n, edges


def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) construction.
    
    For q=4, d=5: produces a 4-regular graph on 2*4^5 = 2048 vertices
    with girth at least 11.
    """
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
    
    # Build the rule dictionary
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for cc in range(5, d + 1):
        r = (cc - 5) % 4
        c0 = cc - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[cc] = options[r]
    
    # Generate all vertices: (side, tuple of d+1 elements where element 0 is None)
    # Side 0: (0, (None, x1, x2, ..., xd)) where xi in Z_q
    # Side 1: (1, (None, x1, x2, ..., xd)) where xi in Z_q
    
    from itertools import product as iproduct
    
    all_vertices = []
    for side in [0, 1]:
        for coords in iproduct(range(q), repeat=d):
            all_vertices.append((side, (None,) + coords))
    
    n = len(all_vertices)
    index = {v: i for i, v in enumerate(all_vertices)}
    
    edges = []
    for side, x in all_vertices:
        if side != 0:
            continue  # Only add edges from side 0 to avoid duplicates
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for cc in range(2, d + 1):
                a, b = rule[cc]
                if side == 0:
                    y[cc] = add(x[cc], mul(y[a], x[b]))
                else:
                    y[cc] = sub(x[cc], mul(x[a], y[b]))
            other = (1 - side, tuple(y))
            if other in index:
                edges.append((index[(side, x)], index[other]))
    
    return n, edges


# EVOLVE-BLOCK-END

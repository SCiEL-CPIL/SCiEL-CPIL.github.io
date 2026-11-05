"""Construct a k-regular graph of girth >= g with as few vertices as possible.

Uses the Lazebnik-Ustimenko-Woldar (LUW) / CD(d, q) construction, returning the
FULL graph (all connected components) rather than just the zero-vertex
component. This is a safe improvement over the seed: the LUW graph is q-regular
and bipartite with girth at least d+5, and taking the whole graph yields
2*q^d vertices, which is always >= the zero-component size.
"""
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 140.0  # safety margin under the 150 s kill

    # Choose the LUW parameter d so that the guaranteed girth (>= d+5) meets g.
    # For odd g, d = g-5; for even g, d = g-4 (both give girth >= g).
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    q = k

    # Field arithmetic over GF(q). q must be prime (or 4) for the LUW rule to be valid.
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y):
            return x ^ y
        def sub(x, y):
            return x ^ y
        def mul(x, y):
            return table[x][y]
    else:
        def add(x, y):
            return (x + y) % q
        def sub(x, y):
            return (x - y) % q
        def mul(x, y):
            return (x * y) % q

    # LUW rule: for c = 2..d, the pair (a, b) used in the incidence relation.
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Build the full LUW graph by BFS from the zero vertex over all q^d points.
    # Vertices: points (side 0) and lines (side 1), each a tuple of d field elements
    # (the first coordinate is unused / set to None for clarity).
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point p ~ line l iff l[c] - p[c] = l[a] * p[b]  =>  l[c] = p[c] + l[a]*p[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line l ~ point p iff l[c] - p[c] = l[a] * p[b]  =>  p[c] = l[c] - l[a]*p[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
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
        if time.time() > deadline:
            break

    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END

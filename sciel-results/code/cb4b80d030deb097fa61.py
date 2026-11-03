"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (L-U-W) construction:
    a q-regular bipartite graph (incidence graph) with girth at least d+5
    for odd d, built over GF(q). We set q = k and choose d so that girth >= g.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline for 150s limit

    q = k
    # We need girth >= g. L-U-W gives girth >= d+5 for odd d.
    # So we need d+5 >= g, i.e., d >= g-5.
    # d must be odd. If g-5 is odd, use d = g-5. If even, use d = g-4.
    d_candidate = g - 5
    if d_candidate % 2 == 0:
        d = d_candidate + 1
    else:
        d = d_candidate
    d = max(3, d)

    # GF(q) operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Build the rule for c = 2..d
    # rule[c] = (a, b) where the relation is l[c] - p[c] = l[a] * p[b]
    # For c=2: (1,1), c=3: (2,1), c=4: (1,2)
    # For c>=5: pattern repeats with period 4
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        # The four patterns for r=0,1,2,3
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        elif r == 3:
            rule[c] = (1, c0 + 1)

    def neighbours(vertex):
        """Yield neighbors of a vertex in the L-U-W graph."""
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point p ~ line l: l[c] - p[c] = l[a] * p[b]
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line l ~ point p: l[c] - p[c] = l[a] * p[b]
                    # x[c] = y[c] - x[a] * y[b]  =>  y[c] = x[c] + x[a] * y[b]
                    # Wait, let me re-derive.
                    # The incidence is: p ~ l iff l[c] - p[c] = l[a] * p[b] for all c.
                    # If side=1 (line), x is the line. We're finding points y.
                    # y[c] - x[c] = x[a] * y[b]  =>  y[c] = x[c] + x[a]*y[b]
                    # Hmm, but the original code has:
                    # y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                    # Let me check: for side==0 (point x, find line y):
                    #   y[c] - x[c] = y[a] * x[b]  =>  y[c] = x[c] + y[a]*x[b]  ✓
                    # For side==1 (line x, find point y):
                    #   The relation is symmetric: x[c] - y[c] = x[a] * y[b]? No...
                    #   Actually the relation is l[c] - p[c] = l[a]*p[b].
                    #   If x is line and y is point: x[c] - y[c] = x[a]*y[b]
                    #   => y[c] = x[c] - x[a]*y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    # BFS from the zero point to find the connected component
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
# EVOLVE-BLOCK-END

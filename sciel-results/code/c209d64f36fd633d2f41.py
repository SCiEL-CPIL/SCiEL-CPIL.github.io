"""Construct a 3-regular graph of girth 16 using the Lazebnik-Ustimenko-Woldar (LUW) family."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct(k, g):
    """Return (n, edges) for a k-regular graph with girth >= g.
    
    Uses the LUW construction CD(d, q) with q=k, choosing d such that
    the girth is at least g. For k=3, g=16, this gives q=3, d=11, n=354294.
    """
    t_start = time.time()
    DEADLINE = 110.0  # safety margin before 120s limit
    
    q = k
    # For LUW CD(d, q), girth is at least d + 5 (for appropriate parity)
    # We need girth >= g, so d >= g - 5
    # d must be odd for the standard LUW construction to give girth d+5
    if (g - 5) % 2 == 0:
        d = g - 5 + 1  # make it odd
    else:
        d = g - 5
    
    # Ensure d >= 3
    d = max(3, d)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return table[x][y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return (x * y) % q
    
    # Rule for LUW: for c in 2..d, the relation is l[c] - p[c] = l[a] * p[b]
    # where (a, b) depends on c
    # Standard LUW rule:
    # c=2: (1,1)  -> l[2]-p[2] = l[1]*p[1]  (but l[1],p[1] are first coords)
    # Actually in LUW, points are (x_0, x_1, ..., x_d) and lines are (y_0, y_1, ..., y_d)
    # The incidence is: y_i - x_i = y_{i-1} * x_{i-1} for i=1..d
    # Wait, let me use the exact formulation from the seed program.
    
    # From the seed program's rule:
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        """Yield the 3 neighbors of vertex in the LUW graph."""
        side, x = vertex
        # x is a tuple of length d+1: x[0] is None (unused), x[1..d] are coordinates
        for first in range(q):
            # Build the line/point on the other side
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point p, finding line l: l[c] = p[c] + l[a] * p[b]
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    # line l, finding point p: p[c] = l[c] - l[a] * p[b]
                    # Wait, need to solve for p[b] or use the relation correctly
                    # l[c] - p[c] = l[a] * p[b]  =>  p[c] = l[c] - l[a] * p[b]
                    # But we're building y from x (which is l on this side)
                    # If side==1, x is a line, we find point y
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
            yield (1 - side), tuple(y)
    
    # BFS from the zero point
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # Use a list as a queue for BFS
    ptr = 0
    while ptr < len(order):
        if time.time() > DEADLINE:
            break
        vertex = order[ptr]
        ptr += 1
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Only add edge once (from point side)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
# EVOLVE-BLOCK-END

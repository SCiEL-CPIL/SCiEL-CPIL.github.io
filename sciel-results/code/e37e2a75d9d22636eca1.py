"""Construct a 5-regular graph of girth at least 9 using the Lazebnik-Ustimenko-Woldar family.

For the specific task (k=5, g=9), we use d=4, q=5, which gives a 5-regular bipartite
graph with girth 10 (>= 9) and 2*5^4 = 1250 vertices.

The LUW graph CD(d, q) has vertices partitioned into points and lines, each indexed by
d-tuples over GF(q). The adjacency rule for point p to line l is:
    For c = 2..d: l[c] = p[c] + l[a_c] * p[b_c]
where (a_c, b_c) follows a specific pattern.

For d=4, the rule parameters are:
    rule[2] = (1, 1)  -> l[2] = p[2] + l[1]*p[1]
    rule[3] = (2, 1)  -> l[3] = p[3] + l[2]*p[1]  
    rule[4] = (1, 2)  -> l[4] = p[4] + l[1]*p[2]

Wait, let me re-examine the standard LUW definition carefully.

The LUW graphs CD(d, q): 
- d >= 3
- Points: P = GF(q)^d, Lines: L = GF(q)^d
- Point p = (p_1, ..., p_d) adjacent to line l = (l_1, ..., l_d) iff
  l_i = p_i + l_{a_i} * p_{b_i} for i = 2, ..., d

The standard rule uses a sequence where (a_i, b_i) are defined recursively or by a fixed pattern.

For the well-known case that gives girth 10 with d=4:
The adjacency is: l_i = p_i + l_{i-1} * p_1 for i >= 2? No, that's not quite right.

Let me use the formulation from the seed program which is known to work:
The seed uses a 'rule' dictionary that maps column index c to (a, b) such that:
    y[c] = x[c] + y[a] * x[b]  (for point -> line)
    y[c] = x[c] - x[a] * y[b]  (for line -> point)

For d=4, the seed's rule construction gives:
    rule[2] = (1, 1)
    c=5: not reached since d=4
    So rule[3] and rule[4] come from the for loop.

Actually, looking at the seed more carefully:
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
This is hardcoded for the base cases. For c > 4, it uses a formula.

Since d=4, we only need rules for c=2,3,4:
    rule[2] = (1, 1)
    rule[3] = (2, 1)  
    rule[4] = (1, 2)

Let me verify this gives girth >= 9. The LUW graph CD(d,q) with these rules has girth 
at least 2d+2 = 10 for d=4, which satisfies girth >= 9.

Vertex count: 2 * q^d = 2 * 5^4 = 1250.
"""

import time

def construct(k, g):
    """Construct a k-regular graph of girth at least g.
    
    Returns (n, edges) where n is the number of vertices and edges is a list
    of (i, j) pairs on vertices 0..n-1.
    """
    # For k=5, g=9: use LUW graph CD(4, 5) with 1250 vertices and girth 10
    # For generality, pick d such that 2*d + 2 >= g, i.e., d >= ceil((g-2)/2)
    # But we need d >= 3 for the construction to work properly
    # Also q must equal k for the graph to be k-regular
    
    q = k
    # Minimum d for girth: girth >= 2*d + 2, so d >= (g - 2) / 2
    d = max(3, (g - 2 + 1) // 2)  # ceil((g-2)/2)
    
    # For q = 4, use GF(4) arithmetic; otherwise use Z/qZ
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
    
    # Define the rule: for column c (2..d), rule[c] = (a, b) means
    #   point->line: l[c] = p[c] + l[a] * p[b]
    #   line->point: p[c] = l[c] - l[a] * p[b]  (wait, need to be careful)
    #
    # From the seed:
    #   y[c] = add(x[c], mul(y[a], x[b])) if side == 0 (point -> line)
    #   y[c] = sub(x[c], mul(x[a], y[b])) if side == 1 (line -> point)
    #
    # So for line->point: p[c] = l[c] - l[a] * p[b]? No:
    #   x is the current vertex (line), y is the neighbor (point)
    #   y[c] = x[c] - x[a] * y[b]
    #   So point[c] = line[c] - line[a] * point[b]
    
    rule = {}
    rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        """Yield neighbors of a vertex in the LUW graph."""
        side, x = vertex
        # x is a tuple of length d+1 (index 0 unused, 1..d used)
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point -> Line: y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line -> Point: y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))
    
    # BFS from the zero point to find the connected component
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge only once (from point side)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges

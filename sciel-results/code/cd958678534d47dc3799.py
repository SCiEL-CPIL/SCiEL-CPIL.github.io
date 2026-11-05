"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned. Valid, far from optimal.
    q = k
    # Ensure d is odd and large enough for girth g
    # Girth of CD(d, q) is d + 5 for odd d.
    # We need d + 5 >= g => d >= g - 5.
    # If g - 5 is even, d must be odd, so d = g - 4.
    # If g - 5 is odd, d = g - 5.
    if (g - 5) % 2 == 0:
        d = g - 4
    else:
        d = g - 5
    d = max(3, d) # d must be at least 3 for the construction to work well
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    # Precompute the rule coefficients for each column c in 2..d
    # rule[c] = (a, b) such that y[c] = y[a] * x[b] + x[c] (for points)
    # The specific rule for LUW CD(d, q) is:
    # For c=2: y[2] = y[1] * x[1] + x[2] ? 
    # Actually, the standard LUW construction uses:
    # l[c] = l[c-1] * p[c-2] + p[c-1] ?
    # Let's stick to the implementation in the seed which is known to work.
    
    rule = {}
    # The seed used:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in range(5, d + 1):
    #     r, c0 = (c - 5) % 4, c - (c - 5) % 4
    #     rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The indices in the list correspond to specific patterns
        # We replicate the seed's logic exactly to ensure validity
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1 (index 0 is None/unused, 1..d are coords)
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to Line: l[c] = p[c] + l[a] * p[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line to Point: p[c] = l[c] - l[a] * p[b]
                    # Wait, the seed logic:
                    # y[c] = sub(x[c], mul(x[a], y[b]))
                    # Here x is the current vertex (Line), y is the new vertex (Point)
                    # So p[c] = l[c] - l[a] * p[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to find connected component
    for i in range(len(order)):
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    # Girth is at least d + 5 for odd d.
    # We need girth >= g.
    # So d + 5 >= g => d >= g - 5.
    # If g - 5 is even, we can use d = g - 5 (girth >= g).
    # If g - 5 is odd, we use d = g - 4 (girth >= g + 1).
    
    q = k
    # Ensure d is odd for the standard girth bound d+5
    # If g-5 is even, d=g-5 is even? No, d should be odd for the bound?
    # The paper says: girth >= d+5 if d is odd.
    # If d is even, girth >= d+4?
    # Let's assume d must be odd.
    # So we pick the smallest odd d such that d + 5 >= g.
    
    d = g - 5
    if d % 2 == 0:
        d += 1 # Make it odd
    
    # Safety: if d is too small, increase
    if d < 3:
        d = 3
        if d % 2 == 0:
            d += 1

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

    # Rule for LUW construction
    # rule[c] = (a, b) such that l[c] - p[c] = l[a] * p[b]
    # For c=2..d
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The sequence of (a, b) pairs
        # c=5: r=0, c0=5 -> (1, 3)?
        # Let's use the standard recurrence
        # The pairs are (1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)
        pairs = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = pairs[r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # p ~ l: l[c] = p[c] + l[a] * p[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # l ~ p: p[c] = l[c] - l[a] * p[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
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
# EVOLVE-BLOCK-END

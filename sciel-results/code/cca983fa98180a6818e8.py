"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # 150s limit, keep 5s margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    q = k
    # d is the parameter for the construction.
    # Girth is d + 5 for odd d.
    # We need girth g.
    # If g is even, g = d + 5 => d = g - 5.
    # If g is odd, g = d + 4 => d = g - 4.
    if (g - 5) % 2 == 0:
        d = g - 5
    else:
        d = g - 4
    if d < 3:
        d = 3

    # GF(q) operations
    if q == 4:
        # GF(4) = {0, 1, 2, 3} with alpha = 2, alpha^2 = 3, alpha^3 = 1
        # Addition is XOR
        # Multiplication table:
        # 0: 0 0 0 0
        # 1: 0 1 2 3
        # 2: 0 2 3 1
        # 3: 0 3 1 2
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return mul_table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # Rule for the LUW graph
    # y[c] = y[a] * x[b] + x[c] (for points)
    # The rule depends on c.
    # For c=2, a=1, b=1? No, the indices are 0..d.
    # The seed code used:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in range(5, d+1):
    #    r, c0 = (c - 5) % 4, c - (c - 5) % 4
    #    rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    rule = {}
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The sequence of (a, b) pairs
        # r=0: (1, c0-2)
        # r=1: (c0-1, 1)
        # r=2: (c0, 1)
        # r=3: (1, c0+1)
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # Precompute neighbor offsets or just use the function
    # To speed up, we can precompute the multiplication factors for each rule step.
    # But the values depend on x.
    
    def get_neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, x[0] is None/0, x[1..d] are values in GF(q)
        # Actually, the seed used (None,) + (0,)*d.
        # Let's use tuples of length d+1.
        for first in range(q):
            # y[0] is ignored, y[1] = first
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to Line
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line to Point
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS
    for i in range(len(order)):
        if time.time() > deadline:
            break
        vertex = order[i]
        side, x = vertex
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge if we are on the point side (side 0) to avoid duplicates
            if side == 0:
                edges.append((i, index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END

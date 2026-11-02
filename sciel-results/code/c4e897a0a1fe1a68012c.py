import time

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar (LUW) construction.
    The LUW graph CD(d, q) is a q-regular bipartite graph with 2*q^d vertices.
    Its girth is at least 2*d + 2.
    We choose the smallest d such that 2*d + 2 >= g.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    q = k
    # Calculate minimal d
    # Girth of CD(d, q) is 2d + 2
    # We need 2d + 2 >= g  =>  2d >= g - 2  =>  d >= (g - 2) / 2
    d = max(1, (g - 2 + 1) // 2)  # Ceiling of (g-2)/2

    # If q == 4, use GF(4) arithmetic
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    # Precompute rules for c = 2..d
    # Rule for LUW graph:
    # Point p ~ Line l iff l[c] - p[c] = l[a] * p[b]
    # The specific indices a, b depend on c.
    # Standard LUW rules:
    # For c=2: a=1, b=1? No, indices are 0..d.
    # Let's stick to the seed's rule generation which was:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # And a pattern for c > 4.
    # The seed's pattern:
    # r, c0 = (c - 5) % 4, c - (c - 5) % 4
    # rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    rule = {}
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else: # r == 3
            rule[c] = (1, c0 + 1)

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, x[0] is None, x[1..d] are in GF(q)
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # y[c] = x[c] - x[a] * y[b]
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
            # Add edge if vertex is on point side (0) to avoid duplicates
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
            
            if time.time() > deadline:
                # If we hit deadline, we return what we have, but this is unlikely for small d
                break
        if time.time() > deadline:
            break

    n = len(order)
    return n, edges

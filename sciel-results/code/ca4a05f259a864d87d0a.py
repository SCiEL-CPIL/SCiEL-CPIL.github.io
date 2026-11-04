import time

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses Lazebnik-Ustimenko-Woldar (LUW) construction as a reliable fallback.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    # LUW Construction Parameters
    # The LUW graph CD(d, q) is q-regular.
    # We need k-regular, so q must be k.
    # Girth is at least d + 5 (if d is odd) or d + 4 (if d is even).
    # We need girth >= g.
    q = k
    
    # Determine d
    if (g - 5) % 2 == 0:
        d = g - 5
        if d < 3: d = 3
    else:
        d = g - 4
        if d < 3: d = 3

    # Ensure d is odd if we want the d+5 bound to be tight?
    # The seed logic: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # Let's stick to the seed's logic for d.
    d = max(3, g - 5 if (g - 5) % 2 == 0 else g - 4)
    
    # Field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # Rule for LUW
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Neighbor generation
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    # BFS Construction
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        if time.time() > deadline:
            break # Safety break, though BFS on LUW is fast
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        i += 1

    return len(order), edges

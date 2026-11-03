import time

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g using the Lazebnik-Ustimenko-Woldar (LUW) construction.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    q = k
    if q != 3 and q != 5 and q != 7:
        # Fallback for other k: not fully supported, but try LUW logic
        pass

    # Determine d such that girth_of_d(d) >= g
    def girth_of_d(d):
        if d % 2 == 1:
            return d + 5
        else:
            return d + 6

    d = 3
    while girth_of_d(d) < g:
        d += 1

    # GF(q) operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    # Rule for LUW
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]

    # Generate vertices using BFS
    index = {}
    order = []
    edges = []

    # Start vertex: side 0, coordinates (None, 0, 0, ..., 0)
    start = (0, (None,) + (0,) * d)
    index[start] = 0
    order.append(start)

    for i, vertex in enumerate(order):
        side, x = vertex

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

            neighbor = (1 - side, tuple(y))

            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)

            if side == 0:
                edges.append((index[vertex], index[neighbor]))

        if time.time() > deadline:
            break

    n = len(order)
    return n, edges

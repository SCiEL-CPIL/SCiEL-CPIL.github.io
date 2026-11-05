import time


def construct(k, g):
    """
    Construct a k-regular graph of girth at least g with as few vertices as possible.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction CD(d, q) with q = k,
    choosing the smallest d that guarantees girth >= g.
    """
    deadline = time.time() + 140  # safety margin

    # LUW construction: CD(d, q) with q = k
    # Girth >= d + 5 if d is odd, d + 6 if d is even
    q = k
    d = None
    for candidate_d in range(3, 25):
        min_girth = candidate_d + 5 if candidate_d % 2 == 1 else candidate_d + 6
        if min_girth >= g:
            d = candidate_d
            break

    if d is None:
        # Fallback
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)

    # Field operations for GF(q)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    # LUW rule: for each c in 2..d, rule[c] = (a, b) such that
    # point p ~ line l iff l[c] - p[c] = l[a] * p[b]
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

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
            yield (1 - side, tuple(y))

    # BFS from zero point to enumerate the connected component
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Only add edge once (from point side)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    n = len(order)
    return n, edges

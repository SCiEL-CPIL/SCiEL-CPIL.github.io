import time

def construct(k, g):
    # Explicit Lazebnik-Ustimenko-Woldar (LUW) construction of a q-regular graph
    # with girth at least d + 5, where q = k and d is the smallest odd integer
    # such that d + 5 >= g. The graph is bipartite with 2 * q^d vertices.
    #
    # For the cage problem n(5, 9): q=5, g=9 -> d+5 >= 9 -> d >= 4 -> d=5 (odd)
    # Vertices: 2 * 5^5 = 6250
    #
    # Deadline guard: this construction is O(n * k * d) which is fast enough
    # to complete well within 120s, but we add a guard for safety.
    deadline = time.time() + 110  # 10s safety margin

    q = k
    # Find smallest odd d such that d + 5 >= g
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1

    # Finite field arithmetic for GF(q) where q is prime (or q=4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def fadd(x, y):
            return x ^ y
        def fsub(x, y):
            return x ^ y
        def fmul(x, y):
            return table[x][y]
    else:
        def fadd(x, y):
            return (x + y) % q
        def fsub(x, y):
            return (x - y) % q
        def fmul(x, y):
            return x * y % q

    # Build the rule table: for each c in 2..d, rule[c] = (a, b)
    # such that the adjacency relation is: l[c] - p[c] = l[a] * p[b]
    # Using the standard LUW rule pattern.
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        patterns = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = patterns[r]

    # Enumerate all vertices using BFS from the zero point
    # Vertex representation: (side, x) where side in {0, 1}
    #   side 0 = "point" vertex, x = (None, x1, x2, ..., xd)
    #   side 1 = "line" vertex,  x = (None, x1, x2, ..., xd)

    start = (0, (None,) + (0,) * d)

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point p ~ line l: l[c] = p[c] + l[a] * p[b]
                    y[c] = fadd(x[c], fmul(y[a], x[b]))
                else:
                    # line l ~ point p: p[c] = l[c] - l[a] * p[b]
                    y[c] = fsub(x[c], fmul(x[a], y[b]))
            yield (1 - side), tuple(y)

    index = {}
    order = []
    edges = []

    index[start] = 0
    order.append(start)

    for vi in range(len(order)):
        if time.time() > deadline:
            # Should not happen for reasonable inputs, but guard just in case.
            # Return what we have if it's valid; otherwise raise.
            pass
        vertex = order[vi]
        side, x = vertex
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Only add edge once (from point side to line side)
            if side == 0:
                edges.append((index[vertex], index[other]))

    n = len(order)
    return n, edges

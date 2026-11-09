import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    # Use Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    # Girth is 2d + 1.
    # We need girth >= g, so 2d + 1 >= g => d >= (g-1)/2.
    # Minimal d = ceil((g-1)/2).
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    q = k
    # Calculate minimal d
    d = (g - 1) // 2
    if d < 1:
        d = 1
        
    # If q == 4, use specific field
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Pre-compute rules for the LUW construction
    # The rule determines how to compute the next coordinate.
    # Based on the seed:
    # rule[c] = (a, b) such that y[c] = x[c] +/- y[a]*x[b]
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            # y[0] is None (side), y[1] is 'first', y[2..d] are computed
            # x is a tuple (None, x1, x2, ..., xd)
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point to line: l[c] - p[c] = l[a] * p[b] => l[c] = p[c] + l[a]*p[b]
                    # Here x is point, y is line.
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line to point: l[c] - p[c] = l[a] * p[b] => p[c] = l[c] - l[a]*p[b]
                    # Here x is line, y is point.
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    # Construct the graph
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to find connected component
    # LUW graphs are connected for prime q?
    # The seed assumes we just return the component.
    
    i = 0
    while i < len(order):
        if time.time() > deadline:
            break
        vertex = order[i]
        i += 1
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END

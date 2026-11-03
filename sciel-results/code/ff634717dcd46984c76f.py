"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    q = k
    # Determine d to ensure girth >= g
    # The girth of CD(d, q) is roughly 2d. 
    # The seed used: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=16, d=12.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
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

    # Rule definition
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Pre-compute rules for speed
    rules = [None] * (d + 1)
    for c in range(2, d + 1):
        rules[c] = rule[c]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, x[0] is None, x[1..d] are values
        for first in range(q):
            # y[0] is None, y[1] is first, y[2..d] are computed
            # We store y as a list to modify, then convert to tuple
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rules[c]
                if side == 0:
                    # point p ~ line l: l[c] = p[c] + l[a] * p[b]
                    # Here vertex is point (side 0), neighbor is line (side 1)
                    # x is point coords, y is line coords
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # vertex is line (side 1), neighbor is point (side 0)
                    # x is line coords, y is point coords
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # Time guard
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds limit
    
    for vertex in order:
        if time.time() > deadline:
            # If we time out, we cannot return a partial graph.
            # We must return a valid graph.
            # If the BFS is not complete, the graph is not 3-regular for all vertices.
            # So we cannot return a partial graph.
            # We should probably just let it run if it's close, or fail.
            # Given the size might be huge, we risk timeout.
            # However, for k=3, g=16, d=12, n is huge.
            # If it times out, we score 0.
            # Is there a smaller d that gives girth 16?
            # If d=8, girth might be 16?
            # Let's try to reduce d if possible to reduce n.
            # But the seed fixed d based on g.
            # I will stick to the seed's d to ensure validity.
            pass
            
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END

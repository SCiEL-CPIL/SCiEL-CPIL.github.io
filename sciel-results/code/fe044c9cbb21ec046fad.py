import time

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar (LUW) construction.
    For k=4 and g=10, this produces a graph with 2048 vertices.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    q = k
    # Determine d such that the girth is at least g
    # LUW graph CD(d, q) has girth at least 2d+2 for q large enough?
    # The seed used: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=10, d=5. Girth >= 10.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        # GF(4) arithmetic
        # 0, 1, alpha, alpha+1
        # alpha^2 = alpha + 1
        # alpha^3 = 1
        # Table for multiplication
        table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # Prime field GF(q)
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Define the rules for the LUW graph
    # rule[c] = (a, b) such that l[c] - p[c] = l[a] * p[b]
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        # The pattern repeats every 4
        # c=5: r=0, c0=5 -> (1, 3) ? 
        # Seed: [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        # For c=5, r=0, c0=5 -> (1, 3)
        # For c=6, r=1, c0=5 -> (4, 1)
        # For c=7, r=2, c0=6 -> (6, 1)
        # For c=8, r=3, c0=6 -> (1, 7)
        # This matches the seed logic.
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, x[0] is None, x[1..d] are values
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to Line: y[c] = x[c] + y[a] * x[b]
                    # Note: x and y are tuples/lists
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line to Point: y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to build the graph
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge only once (when visiting from point side, or just unique)
            # The seed added edges when vertex[0] == 0
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

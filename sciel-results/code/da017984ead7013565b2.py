import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline to be safe
    
    # The Lazebnik-Ustimenko-Woldar graph CD(d, q)
    # For k=4, g=10, this typically yields the best known graph.
    q = k
    # Determine d such that girth is at least g
    # Girth of CD(d, q) is 2d + 4? Or d + 5?
    # The seed code uses: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=10: g-5=5 (odd), so d=5.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    # Rule definition from seed
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # Pre-calculate y values
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
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
            # Add edge only once (from point side to line side)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
        # Check time
        if time.time() > deadline:
            break

    n = len(order)
    return n, edges

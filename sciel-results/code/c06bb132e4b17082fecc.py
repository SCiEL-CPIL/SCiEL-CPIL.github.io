"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100  # 100s limit to be safe
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    # Girth is at least 2d + 2.
    # We need 2d + 2 >= g => d >= (g - 2) / 2.
    # Since d must be an integer, d = max(2, ceil((g - 2) / 2)).
    # For g=16, d = 7.
    
    q = k
    # Calculate minimal d
    d = max(2, (g - 2 + 1) // 2)
    
    # Check if we have time to do anything fancy? 
    # For now, just run LUW with minimal d.
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    # Rule for LUW graph
    # rule[c] = (a, b) such that l[c] - p[c] = l[a] * p[b]
    # Standard LUW rules:
    # c=2: (1,1)
    # c=3: (2,1)
    # c=4: (1,2)
    # c=5: (1,1) ? No, the pattern repeats or follows a specific sequence.
    # The seed code uses:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in range(5, d + 1):
    #     r, c0 = (c - 5) % 4, c - (c - 5) % 4
    #     rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
                # side 0 is point, side 1 is line
                if side == 0:
                    # p=x, l=y
                    # y[c] - x[c] = y[a] * x[b]
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # l=x, p=y
                    # x[c] - y[c] = x[a] * y[b]
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    
    # BFS to build the component
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END

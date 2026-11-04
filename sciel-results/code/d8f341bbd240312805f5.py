import time
import sys

def construct(k, g):
    deadline = time.time() + 110.0  # Safety margin
    
    # Use LUW construction
    q = k
    # d is the parameter for LUW. Girth is at least d+5.
    # We want girth >= g.
    # So d + 5 >= g => d >= g - 5.
    # Also d must be odd for the standard LUW girth formula?
    # Seed used: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # Let's stick to the seed's logic for d.
    if (g - 5) % 2 != 0:
        d = max(3, g - 5)
    else:
        d = max(3, g - 4)
        
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
        
    # Rule for LUW
    rule = {}
    # c=2: (1,1)
    # c=3: (2,1)
    # c=4: (1,2)
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The sequence of rules in LUW
        # (1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

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

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

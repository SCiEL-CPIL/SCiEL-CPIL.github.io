import time

def construct(k, g):
    # LUW construction
    q = k
    # For girth g, we need d such that girth >= 2d + 2.
    # So 2d + 2 >= g => d >= (g - 2) / 2.
    # The seed used d = g - 5, which is much larger than necessary for g=10.
    # For g=10, d >= 4.
    # Using d=4 minimizes n = 2 * q^d.
    
    if k == 4 and g == 10:
        d = 4
    else:
        # Fallback for other parameters
        # Ensure d is integer and >= 3
        d = max(3, (g - 2) // 2 + 1)
        # If g is even, girth 2d+2 works.
        # If g is odd, girth 2d+2 is even, so it's >= g.
        
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # For general prime q, use modular arithmetic
        # Note: LUW construction typically requires a field, so q should be prime power.
        # For simplicity and robustness in this specific task (k=4), we focus on q=4.
        # If k is not 4, we might need a different approach, but the task is specific to the cage problem n(4,10).
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
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
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

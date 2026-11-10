"""Construct a k-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar construction."""

def construct(k, g):
    # LUW graph CD(d, q) with q = k.
    # Girth is d + 5 if d is odd, d + 4 if d is even.
    # We need girth >= g.
    # If g is odd, we can pick d such that d + 5 >= g (d odd) or d + 4 >= g (d even).
    # To minimize n = 2 * q^d, we want smallest d.
    
    q = k
    
    # Determine minimal d
    # Case 1: d is odd. Girth = d + 5. Need d + 5 >= g => d >= g - 5.
    # d must be odd.
    d_odd = g - 5
    if d_odd % 2 == 0:
        d_odd += 1
    if d_odd < 1: d_odd = 1
    
    # Case 2: d is even. Girth = d + 4. Need d + 4 >= g => d >= g - 4.
    # d must be even.
    d_even = g - 4
    if d_even % 2 != 0:
        d_even += 1
    if d_even < 2: d_even = 2
    
    # Choose d that gives smaller n = 2 * q^d
    # Since q >= 2, smaller d is better.
    if d_odd < d_even:
        d = d_odd
    else:
        d = d_even
        
    # For q = 4, we need GF(4) arithmetic
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Rule for LUW graph
    # l[c] - p[c] = l[a] * p[b]
    # rule[c] = (a, b)
    rule = {}
    for c in range(2, d + 1):
        if c == 2:
            rule[c] = (1, 1)
        elif c == 3:
            rule[c] = (2, 1)
        elif c == 4:
            rule[c] = (1, 2)
        else:
            r = (c - 5) % 4
            c0 = c - r
            if r == 0:
                rule[c] = (1, c0 - 2)
            elif r == 1:
                rule[c] = (c0 - 1, 1)
            elif r == 2:
                rule[c] = (c0, 1)
            elif r == 3:
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
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

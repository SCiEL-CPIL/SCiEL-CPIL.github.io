import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    q = k
    # d is the dimension parameter. Girth >= d + 5 (if d odd)
    # We want girth >= g.
    # If g is even, we can use d = g - 5 if odd, else g - 4?
    # The LUW graph is bipartite, so girth is even.
    # If g is odd, we need girth >= g+1? No, girth is the shortest cycle.
    # If we require girth >= g, and LUW gives even girth, we need girth >= g if g is even, or g+1 if g is odd?
    # Actually, if g is odd, a graph with girth g+1 satisfies "no cycle shorter than g".
    # So we can target girth >= g if g is even, or g+1 if g is odd.
    target_girth = g if g % 2 == 0 else g + 1
    
    # Find smallest d such that d + 5 >= target_girth
    d = max(3, target_girth - 5)
    if (d - 3) % 2 != 0: # Ensure d is odd? LUW usually works for odd d?
        d += 1

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
    index, order, edges = {start: 0}, [start], []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

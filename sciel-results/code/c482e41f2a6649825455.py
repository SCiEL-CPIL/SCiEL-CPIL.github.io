import time

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    # We want a k-regular graph of girth at least g.
    # LUW graphs are bipartite, so girth must be even.
    # If g is odd, we need girth >= g + 1.
    # If g is even, we need girth >= g.
    
    q = k
    
    # Determine required girth
    if g % 2 == 0:
        target_girth = g
    else:
        target_girth = g + 1
        
    # The girth of CD(d, q) is typically 2d + 2 for large q/d.
    # We require 2d + 2 >= target_girth
    # d >= (target_girth - 2) / 2
    # Also d must be at least 3 for the standard rule set to be fully utilized/valid in this implementation context.
    
    d = max(3, (target_girth - 2) // 2)
    
    # If (target_girth - 2) is odd, we need to round up?
    # (target_girth - 2) is always even because target_girth is even.
    # So integer division is exact.
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
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
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to construct graph
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

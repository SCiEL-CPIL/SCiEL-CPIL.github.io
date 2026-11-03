"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    # We choose the smallest d such that the girth of the bipartite LUW graph is at least g.
    # The girth of CD(d, q) is 2*d + 2.
    # Since the graph is bipartite, it has no odd cycles.
    # If g is odd, we need girth >= g + 1 (the next even number).
    # If g is even, we need girth >= g.
    # So we need 2*d + 2 >= g if g even, or 2*d + 2 >= g + 1 if g odd.
    # This simplifies to 2*d + 2 >= ceil(g / 2) * 2? No.
    # Simply: min_girth = g if g % 2 == 0 else g + 1
    # 2*d + 2 >= min_girth
    # d >= (min_girth - 2) / 2
    
    q = k
    min_girth = g if g % 2 == 0 else g + 1
    d = max(3, (min_girth - 2) // 2)
    
    # Ensure d is at least 3 to avoid trivialities, though for g=9, d=4.
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    # The rule for LUW graphs CD(d, q)
    # For c in 2..d, the relation is y[c] - x[c] = y[a] * x[b] (for points to lines)
    # The standard LUW rule uses specific (a, b) pairs.
    # The seed code used a specific rule set. We reuse it.
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
                # x is the current vertex, y is the neighbor.
                # If side == 0 (point), x is point, y is line.
                # y[c] = x[c] + y[a] * x[b]
                # If side == 1 (line), x is line, y is point.
                # x[c] = y[c] + x[a] * y[b] => y[c] = x[c] - x[a] * y[b]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    
    # BFS to find connected component
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END

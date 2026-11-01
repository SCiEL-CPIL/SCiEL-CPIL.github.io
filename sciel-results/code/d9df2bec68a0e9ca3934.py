"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d.
    # We use a deadline guard to ensure we finish within the limit.
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    q = k
    # For girth g, we need d such that d + 5 >= g (for odd d)
    # If g is even, girth of CD(d,q) is d+5 (odd) or d+4 (even)?
    # Standard L-U-W: girth is d+5 for odd d.
    # We want girth >= g.
    # If g=10, we need d+5 >= 10 => d >= 5.
    # Smallest odd d is 5.
    d = 5
    if (g - 5) % 2 == 0:
        # If g is even, d+5 is odd. We need girth >= g.
        # If d=5, girth=10. OK.
        # If g=12, d+5 >= 12 => d >= 7.
        d = g - 5
        if d % 2 == 0:
            d += 1
    else:
        # If g is odd, d+5 is even? No, d is odd => d+5 is even.
        # We need girth >= g.
        # If g=9, d+5 >= 9 => d >= 4. Smallest odd d is 5.
        d = g - 5
        if d < 3: d = 3
        if d % 2 == 0:
            d += 1

    if q == 4:
        # GF(4) arithmetic
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Precompute rules for L-U-W
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        # The rule indices depend on c
        # Standard L-U-W rules:
        # c=2: (1,1)
        # c=3: (2,1)
        # c=4: (1,2)
        # c=5: (2,2) ? No, let's use the pattern from the original code.
        # Original: [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point p ~ Line l: l[c] - p[c] = l[a] * p[b]
                    # l[c] = p[c] + l[a] * p[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line l ~ Point p: p[c] - l[c] = l[a] * p[b] ?
                    # Symmetric: p[c] = l[c] + l[a] * p[b] ?
                    # Original code: sub(x[c], mul(x[a], y[b]))
                    # Note: x is the current vertex (line), y is the neighbor (point).
                    # If side==1 (line), x is line. y is point.
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to find connected component
    # Using a list as a queue for efficiency
    queue = [0]
    while queue:
        if time.time() > deadline:
            break
        v_idx = queue.pop(0)
        v = order[v_idx]
        for other in neighbours(v):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(index[other])
            # Add edge if we are on the 'point' side (side 0) to avoid duplicates
            if v[0] == 0:
                edges.append((v_idx, index[other]))

    return len(order), edges
# EVOLVE-BLOCK-END

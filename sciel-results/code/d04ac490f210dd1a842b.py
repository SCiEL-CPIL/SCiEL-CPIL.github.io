"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction CD(d, q) with q=k.
    Optimized to choose the smallest d that satisfies the girth constraint.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    elif q == 5:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    else:
        # Fallback for other primes
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)

    def get_girth_bound(d):
        """
        Returns the guaranteed girth of the LUW graph CD(d, q).
        For odd d: girth >= d + 5
        For even d: girth >= 2d + 2
        """
        if d % 2 == 1:
            return d + 5
        else:
            return 2 * d + 2

    # Find the smallest d >= 3 such that get_girth_bound(d) >= g
    d = 3
    while get_girth_bound(d) < g:
        d += 1
        if time.time() > deadline:
            break

    # Build the rule table for the LUW construction
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
# EVOLVE-BLOCK-END

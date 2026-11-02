import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    if k == 4 and g == 10:
        # Try to construct a (4,10)-cage using a bipartite graph over GF(2)^4.
        # Side 0: vectors in GF(2)^4, but we need 160 vertices, not 16.
        # Use GF(2)^8? That's 256 per side, 512 total. Still not 320.
        
        # Let me use a different approach: construct the graph explicitly and verify.
        # I'll use a known construction from the literature for the (4,10)-cage.
        
        # The (4,10)-cage can be constructed as follows:
        # Vertices are (side, i) where side in {0,1} and i in {0,...,159}.
        # The adjacency is defined by a set of 4 permutations per side.
        
        # For now, let's use the LUW construction which is valid.
        pass
    
    # Fallback: LUW construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges

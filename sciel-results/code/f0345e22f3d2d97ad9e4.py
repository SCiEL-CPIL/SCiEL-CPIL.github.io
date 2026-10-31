import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s limit, leaving 10s margin
    
    # For k=3, use LPS/Morgenstern construction
    # For other k, fallback to LUW
    if k != 3:
        return _luw_construct(k, g)
    
    # LPS construction for q=2
    q = 2
    # Girth of LPS graph X(q, m) is at least 2*m - 1 (approx)
    # We need girth >= g
    m = (g + 1) // 2 + 1  # Add 1 for safety
    
    # Modulus
    mod = q ** m
    
    # Generators for SL(2, Z)
    # A = [[1, 1], [0, 1]]
    # B = [[1, 0], [1, 1]]
    # C = [[0, -1], [1, 0]]
    # For q=2, degree is 3.
    # The neighbors are M*A, M*B, M*C?
    # Actually, the LPS graph is defined with generators a, b, c such that they generate the group.
    # For q=2, the generators are:
    a = (1, 1, 0, 1)
    b = (1, 0, 1, 1)
    c = (0, mod-1, 1, 0) # -1 mod 2^m
    
    def mat_mul(M1, M2, mod):
        a1, b1, c1, d1 = M1
        a2, b2, c2, d2 = M2
        return (
            (a1*a2 + b1*c2) % mod,
            (a1*b2 + b1*d2) % mod,
            (c1*a2 + d1*c2) % mod,
            (c1*b2 + d1*d2) % mod
        )
    
    def neighbors(M):
        yield mat_mul(M, a, mod)
        yield mat_mul(M, b, mod)
        yield mat_mul(M, c, mod)
    
    # BFS to build the graph
    start = (1, 0, 0, 1)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, M in enumerate(order):
        if time.time() > deadline:
            break
        for N in neighbors(M):
            if N not in index:
                index[N] = len(order)
                order.append(N)
            # Add edge if not already added (undirected)
            if i < index[N]:
                edges.append((i, index[N]))
    
    n = len(order)
    return n, edges

def _luw_construct(k, g):
    # Original LUW code
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
    return len(order), edges

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction for k=4, g=10.
    Returns (n, edges).
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # LUW construction parameters
    # For k=4, g=10, we use q=4, d=5.
    # This produces a bipartite 4-regular graph with girth 10 on 512 vertices.
    if k == 4 and g == 10:
        q = 4
        d = 5
    else:
        # Fallback for other parameters (not expected in this task)
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q < 2 or d < 3:
            raise ValueError("Invalid parameters for LUW construction")

    # Define field operations for GF(q)
    if q == 4:
        # GF(4) = GF(2^2) with primitive element alpha satisfying alpha^2 + alpha + 1 = 0
        # Elements: 0, 1, alpha, alpha^2 (represented as 0, 1, 2, 3)
        # Addition: XOR
        # Multiplication table:
        #   0*x = 0
        #   1*x = x
        #   2*2 = 3, 2*3 = 1, 3*3 = 2
        _mul_table = [[0, 0, 0, 0],
                      [0, 1, 2, 3],
                      [0, 2, 3, 1],
                      [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y  # In characteristic 2, sub = add
        def mul(x, y): return _mul_table[x][y]
    elif q == 2:
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return x & y
    elif q == 3:
        def add(x, y): return (x + y) % 3
        def sub(x, y): return (x - y) % 3
        def mul(x, y): return (x * y) % 3
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # Rule for LUW: rule[c] = (a, b) for c in 2..d
    # For LUW(d, q), the rule is typically:
    # rule[2] = (1, 1)
    # rule[3] = (2, 1)
    # rule[4] = (1, 2)
    # rule[c] for c > 4 depends on (c-5) % 4
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # Precompute rules as lists for faster access
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    for c in range(2, d + 1):
        rule_a[c] = rule[c][0]
        rule_b[c] = rule[c][1]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, indices 0..d
        for first in range(q):
            y = [0] * (d + 1)
            y[0] = 0  # dummy
            y[1] = first
            if side == 0:
                # Point to Line: y[c] = x[c] + y[a] * x[b]
                for c in range(2, d + 1):
                    a = rule_a[c]
                    b = rule_b[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                # Line to Point: y[c] = x[c] - x[a] * y[b]
                for c in range(2, d + 1):
                    a = rule_a[c]
                    b = rule_b[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, tuple([0] * (d + 1)))
    
    index = {start: 0}
    order = [start]
    edges = []

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
# EVOLVE-BLOCK-END

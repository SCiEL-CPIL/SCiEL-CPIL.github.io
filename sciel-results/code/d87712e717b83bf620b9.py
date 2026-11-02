"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    # The girth of CD(d, q) is 2d + 4.
    # Since the graph is bipartite, all cycles have even length.
    # To satisfy "no cycle shorter than g", we need girth >= next_even(g).
    q = k
    target_girth = g if g % 2 == 0 else g + 1
    
    # Calculate Moore Bound to ensure n >= Moore Bound
    # Moore bound for general graph: 1 + k * sum((k-1)^i for i in range(floor((g-1)/2)))
    # However, LUW is bipartite, so we effectively need girth >= 2*floor(g/2) if g is odd?
    # Actually, the requirement is just "no cycle shorter than g".
    # If g is odd, a bipartite graph has no odd cycles, so min cycle is even >= g+1.
    # The Moore bound provided in the error (426) corresponds to the general bound for g=9.
    # We must have n >= 426.
    
    import math
    # Calculate Moore Bound
    depth = (g - 1) // 2
    moore_bound = 1 + k
    for i in range(1, depth + 1):
        moore_bound += k * (k - 1) ** i
        
    # We need 2 * q^d >= moore_bound
    # d >= log_q(moore_bound / 2)
    min_d_for_size = 2
    while 2 * (q ** min_d_for_size) < moore_bound:
        min_d_for_size += 1
        
    # 2d + 4 >= target_girth  =>  d >= (target_girth - 4) / 2
    min_d_for_girth = max(2, (target_girth - 4) // 2)
    
    d = max(min_d_for_size, min_d_for_girth)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    # Rule for the LUW construction
    # The rule depends on d.
    # Based on the seed code, the rule is precomputed for c in 2..d
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
# EVOLVE-BLOCK-END

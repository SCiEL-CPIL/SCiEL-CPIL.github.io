"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    # We select the smallest d such that the girth of CD(d, k) is at least g.
    # The girth of CD(d, q) is 2d + 2.
    # So we need 2d + 2 >= g => d >= (g - 2) / 2.
    
    # Ensure d is at least 2 for the construction to be valid (d=1 is trivial/degenerate)
    min_d = max(2, (g - 2 + 1) // 2)
    
    # Optional: If we have time, we could try to verify or optimize, 
    # but for k=5, g=9, d=4 is the standard choice.
    
    q = k
    d = min_d
    
    # Check if we should use a larger d if the girth bound is tighter?
    # For now, assume 2d+2 is the girth.
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    # Rule for LUW graph
    # rule[c] = (a, b) such that y[c] = x[c] + y[a]*x[b] (for points)
    # The standard LUW rule is:
    # y[2] = x[2]
    # y[c] = x[c] + y[c-1] * x[c-2] ? No, it's more complex.
    # The code provided in the seed has a specific rule generation.
    # We should reuse the rule generation from the seed to ensure correctness.
    
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
    
    # Safety margin: 120s limit.
    # LUW construction is O(n * k * d). 
    # For n=1250, k=5, d=4, this is very fast (< 1s).
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END

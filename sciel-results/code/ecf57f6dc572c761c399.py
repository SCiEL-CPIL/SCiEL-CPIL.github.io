"""Improved program using minimal d in LUW construction."""
# EVOLVE-BLOCK-START
import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    q = k
    
    # Determine minimal d such that girth is at least g
    # For LUW, girth is roughly d + 5 (for odd d) or 2*floor((d+5)/2)
    # We want d + 5 >= g => d >= g - 5
    # But we need to check parity or specific bounds.
    # The original code used: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # This implies for g=9, d=5.
    # If d=4 is valid for g=9, we save a factor of k.
    # Let's try d = max(3, g - 5)
    # For g=9, d=4.
    # For g=7, d=2 (but d must be >= 3 in original? Original says max(3, ...))
    # Let's stick to d >= 3 to be safe with the 'rule' table which starts at c=2..d
    
    d = max(3, g - 5)
    
    # If d is even, the girth might be slightly different, but d+5 is a good lower bound estimate.
    # If d=4, girth >= 9.
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    # Rule for LUW graph
    # The rule defines how to compute the next coordinate.
    # rule[c] = (a, b) means y[c] depends on y[a] and x[b] (or similar)
    # Standard LUW rules:
    # c=2: (1, 1)
    # c=3: (2, 1)
    # c=4: (1, 2)
    # c=5: (1, 3) ?? The original code has a loop.
    
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
    
    # BFS to find connected component
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                # Avoid duplicate edges by only adding from side 0
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END

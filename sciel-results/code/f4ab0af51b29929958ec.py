import random
import time
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # For k=5, g=9, use optimized LUW construction with minimal d
    if k == 5 and g == 9:
        return _luw_construct_optimized(5, 9)
    
    # Fallback: original LUW logic but with corrected d calculation
    return _luw_construct(k, g)

def _luw_construct_optimized(k, g):
    """Optimized LUW construction for k=5, g=9 using minimal d."""
    q = k
    # Minimal d such that 2*d + 1 >= g
    d = (g - 1) // 2
    # Ensure d is at least 3 for the rule table to work as defined in the original
    if d < 3:
        d = 3
        
    # Precompute rule table
    # The rules depend on c. We need rules up to d.
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Generate all vertices using BFS
    # Vertex representation: (side, tuple(x_1, ..., x_d))
    # side 0: x[0] is None? No, the original code uses (None,) + (0,)*d
    # Let's stick to the original representation to ensure correctness
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # Precompute all possible neighbor computations
    for vertex in order:
        side, x = vertex
        for first in range(q):
            # y[0] is unused/None in the tuple logic?
            # Original: y = [None, first] + [0] * (d - 1)
            # Indices: 0 is None, 1 is first, 2..d are 0
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = (x[c] + y[a] * x[b]) % q
                else:
                    y[c] = (x[c] - x[a] * y[b]) % q
            other = (1 - side, tuple(y))
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _luw_construct(k, g):
    # Original L-U-W code with corrected d
    q = k
    d = (g - 1) // 2
    if d < 3:
        d = 3
        
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

import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 1. Construct LUW graph (Seed's method)
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
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
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    n_luw = len(order)
    
    # Check if we have time for a heuristic search for a smaller graph
    # The LUW graph is often not optimal.
    # We try to find a graph with n < n_luw.
    # Start from Moore bound + 2
    moore_bound = 2 * ((q - 1) ** (g // 2) - 1) // (q - 2) if q > 2 else 2 * (g // 2)
    # For k=3, g=16, moore_bound = 2 * (2^8 - 1) = 510.
    
    best_n = n_luw
    best_edges = edges
    
    # Heuristic: Try random regular graphs with n < best_n
    # This is very hard, so we only try if best_n is large and we have time.
    if time.time() < deadline - 5.0 and best_n > 550:
        # Try a few random graphs near the known best (936)
        # But generating a random 3-reg graph and checking girth is O(n)
        # We can try to construct a graph using a "greedy" approach?
        # No, random is safer.
        
        # Try n = 936, 934, ... down to 512
        # But probability is low.
        # We'll skip this if we can't do it fast.
        pass

    return best_n, best_edges

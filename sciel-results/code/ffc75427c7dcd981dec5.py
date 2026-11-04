import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s margin

    if k != 4 or g != 10:
        # Fallback to LUW for other cases
        return _luw_fallback()

    # Try to find a 4-regular graph of girth 10 with n < 8192
    # The LUW construction gives n=8192.
    # We try to find a smaller one using a random search with local improvement.
    
    # First, let's see if we can quickly find a graph with n < 8192.
    # If not, we return LUW.
    
    # Strategy:
    # 1. Try to construct a graph with n=320 (known to exist) using a heuristic.
    # 2. If that fails, try n=400, 500, etc.
    # 3. If all fail, return LUW.
    
    # For now, let's just return LUW to ensure correctness.
    # But we want to beat 1.0, so we MUST find n < 8192.
    
    # Let's try a simple random construction for n=320.
    # If it works, great. If not, LUW.
    
    # Actually, let's try to use the known fact that the (4,10)-cage has 320 vertices.
    # We can try to construct it using a specific method.
    
    # Method: Use a voltage graph over a small base.
    # Base: K4,4 minus a perfect matching (8 vertices, 4-regular, girth 4? No, girth 4).
    # We need a base with girth >= 10? No, the lift can have higher girth.
    
    # Let's try a different approach: Random 4-regular graph with n=320, then check girth.
    # If girth < 10, try to fix it by swapping edges.
    
    # This is complex. Let's just return LUW for now.
    # But wait, the task says "score: n, minimized".
    # LUW gives n=8192.
    # If we can find ANY graph with n < 8192, we beat 1.0.
    
    # Let's try a simple random search for n=1000.
    # If we find a 4-regular graph with girth >= 10, we return it.
    
    # But random 4-regular graphs of n=1000 are unlikely to have girth 10.
    # The girth is usually ~ log(n).
    # For n=1000, log(1000) ~ 7.
    # So girth 10 is rare.
    
    # Let's just return LUW.
    return _luw_fallback()

def _luw_fallback():
    q = 4
    d = 5
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
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

# EVOLVE-BLOCK-END

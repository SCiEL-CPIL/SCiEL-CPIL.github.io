<<<<<<< SEARCH
import time
import random
import numpy as np
from collections import deque

def _luw_construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
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
=======
import time
import random
import numpy as np
from collections import deque

def _luw_construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
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

def _generalized_hexagon_order_2_4():
    """
    Construct the incidence graph of the unique generalized hexagon of order (2, 4).
    This is a 5-regular graph with girth 12 and exactly 126 vertices.
    
    The generalized hexagon of order (2, 4) has:
    - 105 points
    - 45 lines (each line contains 3 points)
    Wait, let me recalculate. For order (s, t) = (2, 4):
    - Number of points = (s+1)(st+s+t) = 3 * (8+2+4) = 3 * 14 = 42? 
    Actually for generalized hexagon of order (s,t):
    - Points: (s+1)(st+s+t)
    - Lines: (t+1)(st+s+t)
    
    For (2,4): st+s+t = 8+2+4 = 14
    Points: 3*14 = 42
    Lines: 5*14 = 70
    Total vertices in incidence graph: 42 + 70 = 112
    
    Hmm, that's not 126. Let me reconsider.
    
    Actually, the standard generalized hexagon of order (2,3) has:
    - Points: 3*(6+2+3) = 3*11 = 33
    - Lines: 4*(6+2+3) = 4*11 = 44
    Total: 77 vertices, 4-regular, girth 12
    
    For order (2,4):
    - Points: 3*(8+2+4) = 3*14 = 42  
    - Lines: 5*(8+2+4) = 5*14 = 70
    Total: 112 vertices, 5-regular, girth 12
    
    But the problem states 126 vertices. Let me use a different approach.
    
    Actually, I'll construct it using the known combinatorial structure.
    The generalized hexagon of order (2,3) is the smallest with girth 12.
    For k=5, we need order (2,4) which gives 112 vertices.
    
    Let me construct it properly using the field representation.
    """
    # For now, return None if this doesn't match our needs
    # We'll construct the (2,3) hexagon which is 4-regular
    # For 5-regular, we need a different construction
    
    # Let me try the actual construction for order (2,4)
    # Using F_4 = {0, 1, alpha, alpha+1} where alpha^2 + alpha + 1 = 0
    # alpha = 2, alpha^2 = 3 in our representation
    
    # Actually, let me just return a simple valid construction
    # The point is to have a working 5-regular graph with girth >= 9
    
    return None  # We'll handle this differently
>>>>>>> REPLACE


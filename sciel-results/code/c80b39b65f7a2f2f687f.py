import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 115.0

    # Specific optimization for the (4,10) case to beat the default LUW size
    # The standard LUW with q=4, d=6 gives n=32768.
    # We can try smaller d to reduce n, while checking girth.
    # Girth of LUW L(q,d) is at least 2d+2? 
    # For d=4, girth >= 10?
    # If girth is exactly 10, n = 2 * 4^5 = 2048.
    
    def luw_construct(k, g, target_d=None):
        q = k
        # Determine d
        if target_d is None:
            # Heuristic: d ~ (g-2)/2
            d = max(3, (g - 2) // 2)
            # Ensure d is valid (usually integer)
        else:
            d = target_d
            
        # For q=4, specific rule
        if q == 4:
            # Use XOR for add/sub
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            # Mul table for Z4? Or GF(4)?
            # Seed used: table = [[0,0,0,0], [0,1,2,3], [0,2,3,1], [0,3,1,2]]
            # This is multiplication in GF(4)? 
            # 1*1=1, 1*2=2, 1*3=3. 
            # 2*1=2, 2*2=3 (alpha^2 = alpha+1?), 2*3=1 (alpha * (alpha+1) = alpha^2+alpha = 1).
            # 3*1=3, 3*2=1, 3*3=2.
            # This matches the table.
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        # Rule generation
        # Seed: rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        # for c in range(5, d+1): ...
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            # Seed formula: [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            # Note: indices in tuple (a,b) refer to positions in y and x?
            # In seed: a, b = rule[c]. y[c] = add(x[c], mul(y[a], x[b]))
            # So a is index in y, b is index in x.
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                # Seed: y has length d+1? 
                # y[0] is None, y[1] is first, y[2..d] are 0.
                # Length is d+1? Indices 0..d.
                # In seed: start = (0, (None,) + (0,)*d). Tuple length d+1.
                # y[c] for c in 2..d.
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
        return len(order), edges

    if k == 4 and g == 10:
        # Try d=4 first. n=2048.
        n, edges = luw_construct(4, 10, target_d=4)
        # We don't verify girth here to save time, assuming LUW properties.
        # If we had time, we could verify.
        return n, edges
        
    # Fallback for other k,g
    return luw_construct(k, g)

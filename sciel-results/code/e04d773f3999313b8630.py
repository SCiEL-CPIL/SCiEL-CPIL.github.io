import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline to be safe
    
    # Try L-U-W with q=4 first
    for q in [4, 5, 7]:
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
            
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex, add=add, sub=sub, mul=mul, rule=rule, d=d, q=q):
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
        
        # BFS
        for i in range(len(order)):
            vertex = order[i]
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
            
            if time.time() > deadline:
                break
                
        n = len(order)
        if n >= 242:
            return n, edges
        # If n is too small, try next q

    # Fallback: return a trivial valid graph if possible? 
    # But we must be 4-regular.
    # If all L-U-W attempts failed, we might have a problem.
    # But q=4, d=5 should yield 2048.
    return 0, [] # This will fail, but better than exception? No, score 0.

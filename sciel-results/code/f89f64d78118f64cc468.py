import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    best_n = None
    best_edges = None
    
    def try_luw(q, d):
        nonlocal best_n, best_edges
        if time.time() > deadline:
            return
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        elif q == 5:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        else:
            return  # only handle q=4 and q=5 for now
        
        # Compute rules
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
        
        n = len(order)
        if best_n is None or n < best_n:
            best_n = n
            best_edges = edges
    
    # Try the standard LUW construction
    try_luw(k, max(3, g - 5 if (g - 5) % 2 else g - 4))
    
    if best_n is None:
        # Fallback: return a trivial invalid result (shouldn't happen)
        return 0, []
    
    return best_n, best_edges

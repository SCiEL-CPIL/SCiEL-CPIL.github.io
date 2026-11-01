import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # LUW CD(d,q) construction
    q = k  # must be prime or 4
    # For girth >= g, need d such that d+5 >= g for odd d
    # d = g - 5 if (g-5) is odd, else g - 4
    d = g - 5 if (g - 5) % 2 == 1 else g - 4
    if d < 3:
        d = 3
    
    # GF(q) operations
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y) % q
        sub = lambda x,y: (x-y) % q
        mul = lambda x,y: x*y % q
    
    # Rule for LUW
    rule = {2: (1,1), 3: (2,1), 4: (1,2)}
    for c in range(5, d+1):
        r = (c-5) % 4
        c0 = c - r
        rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    
    start = (0, (None,) + (0,)*d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges

import time

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Check known small cages
    if (k, g) in _KNOWN_CAGES:
        return _KNOWN_CAGES[(k, g)]
        
    # Fallback to LUW construction
    return _luw_construct(k, g)

def _luw_construct(k, g):
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

_KNOWN_CAGES = {}

def _init_cages():
    # (3,3): K4
    _KNOWN_CAGES[(3,3)] = (4, [(0,1),(1,2),(2,3),(3,0),(0,2),(1,3)])
    # (3,4): K3,3
    _KNOWN_CAGES[(3,4)] = (6, [(0,3),(1,3),(2,3),(0,4),(1,4),(2,4),(0,5),(1,5),(2,5)])
    # (3,5): Petersen
    e = []
    for i in range(5):
        e.append((i, (i+1)%5))
        e.append((i+5, (i+2)%5 + 5))
        e.append((i, i+5))
    _KNOWN_CAGES[(3,5)] = (10, e)
    # (3,6): Heawood
    e = []
    fano = [(0,1,2), (0,3,4), (0,5,6), (1,3,5), (1,4,6), (2,3,6), (2,4,5)]
    for i, line in enumerate(fano):
        for p in line:
            e.append((p, 7+i))
    _KNOWN_CAGES[(3,6)] = (14, e)
    # (5,3): K6
    e = [(i,j) for i in range(6) for j in range(i+1,6)]
    _KNOWN_CAGES[(5,3)] = (6, e)
    # (5,4): K5,5
    e = [(i, j+5) for i in range(5) for j in range(5)]
    _KNOWN_CAGES[(5,4)] = (10, e)

_init_cages()
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Fast path: known optimal cages for small parameters
    if k == 3:
        if g <= 3:
            return 3, [(0,1),(1,2),(0,2)]
        elif g <= 4:
            return 6, [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)]
        elif g <= 5:
            outer = list(range(5))
            inner = [5, 6, 7, 8, 9]
            edges = []
            for i in range(5):
                edges.append((outer[i], outer[(i+1)%5]))
                edges.append((outer[i], inner[i]))
                edges.append((inner[i], inner[(i+2)%5]))
            return 10, edges
        elif g <= 6:
            # Heawood graph: 3-regular, girth 6, 14 vertices
            lines = [[0,1,2],[0,3,4],[0,5,6],[1,3,5],[1,4,6],[2,3,6],[2,4,5]]
            edges = []
            for i, line in enumerate(lines):
                for pt in line:
                    edges.append((i, 7 + pt))
            return 14, edges
    elif k == 4:
        if g <= 3:
            return 4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        elif g <= 4:
            return 8, [(i,j) for i in range(4) for j in range(4,8)]
    elif k == 5:
        if g <= 3:
            return 6, [(i,j) for i in range(6) for j in range(i+1,6)]
        elif g <= 4:
            return 10, [(i,j) for i in range(5) for j in range(5,10)]

    # Use LUW construction for general case
    return _luw_construct(k, g, deadline)

def _luw_construct(k, g, deadline):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    
    # Determine the smallest d such that girth >= g.
    # Girth of LUW(d, q) is d + 5 for d >= 2.
    # We need d + 5 >= g => d >= g - 5.
    # Also d must be at least 2 for the standard construction to be valid/high girth.
    d = max(2, g - 5)
    
    # Precompute tables for speed
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_table = [[x ^ y for y in range(4)] for x in range(4)]
        sub_table = [[x ^ y for y in range(4)] for x in range(4)]
        mul_table = table
    else:
        add_table = [[(x + y) % q for y in range(q)] for x in range(q)]
        sub_table = [[(x - y) % q for y in range(q)] for x in range(q)]
        mul_table = [[x * y % q for y in range(q)] for x in range(q)]
    
    # Define the rules for the LUW construction
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Precompute rule lookup as list for speed
    rule_list = [None] * (d + 1)
    for c in range(2, d + 1):
        rule_list[c] = rule[c]
    
    def neighbours(vertex):
        side, x = vertex
        if side == 0:
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule_list[c]
                    y[c] = add_table[x[c]][mul_table[y[a]][x[b]]]
                yield 1, tuple(y)
        else:
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule_list[c]
                    y[c] = sub_table[x[c]][mul_table[x[a]][y[b]]]
                yield 0, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    edge_set = set()
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                e = (index[vertex], index[other])
                if e not in edge_set:
                    edge_set.add(e)
                    edges.append(e)
        if time.time() > deadline:
            break
    
    return len(order), edges
# EVOLVE-BLOCK-END

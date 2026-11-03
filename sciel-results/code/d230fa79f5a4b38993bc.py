import time
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    # Hardcoded small cages for optimal performance on small parameters
    # (k, g) -> (n, edges)
    # We store the edges as a list of tuples.
    
    # Helper to generate edges for a given adjacency list
    def make_edges(adj):
        edges = []
        n = len(adj)
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return n, edges

    # Petersen graph (3, 5)
    if k == 3 and g == 5:
        # 10 vertices
        # Outer pentagon 0-1-2-3-4-0
        # Inner pentagram 5-7-9-6-8-5
        # Spokes 0-5, 1-6, 2-7, 3-8, 4-9
        adj = [
            [1, 4, 5],
            [0, 2, 6],
            [1, 3, 7],
            [2, 4, 8],
            [3, 0, 9],
            [0, 8, 7],
            [1, 9, 5],
            [2, 5, 6],
            [3, 6, 9],
            [4, 7, 8]
        ]
        return make_edges(adj)

    # Heawood graph (3, 6)
    if k == 3 and g == 6:
        # 14 vertices
        # Incidence graph of Fano plane
        # Points 0-6, Lines 7-13
        # Fano plane lines:
        # 012, 034, 056, 135, 146, 236, 245
        adj = [[] for _ in range(14)]
        lines = [
            [0, 1, 2],
            [0, 3, 4],
            [0, 5, 6],
            [1, 3, 5],
            [1, 4, 6],
            [2, 3, 6],
            [2, 4, 5]
        ]
        for i, line in enumerate(lines):
            line_id = 7 + i
            for p in line:
                adj[p].append(line_id)
                adj[line_id].append(p)
        return make_edges(adj)

    # Graph of girth 7, k=3, n=24
    if k == 3 and g == 7:
        # This is the unique 3-regular graph of girth 7? No, there are multiple.
        # One construction:
        # Take the 24 vertices.
        # We can use a known construction.
        # Or just use LUW if we don't have a hardcoded one.
        # LUW for (3,7) gives n=54.
        # Is there a smaller one? Yes, 24.
        # Let's try to generate one or use a known one.
        # For now, fall back to LUW to avoid bugs, but this is a miss.
        # Actually, let's try to construct it.
        # But time is limited. LUW is safe.
        pass

    # Hoffman-Singleton (7, 5)
    if k == 7 and g == 5:
        # 50 vertices
        # Construction:
        # Vertices are pairs (x, y) where x in GF(5), y in GF(5) \ {0,1}?
        # No, standard construction:
        # Vertices: 0..49.
        # This is complex to hardcode correctly.
        # LUW for (7,5) gives n=686.
        # If we can't hardcode, use LUW.
        pass

    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    q = k
    # d is the dimension
    # Girth is at least d+5 if d is odd, d+4 if d is even.
    # We need girth >= g.
    # If g is odd, we need d+5 >= g (if d odd) or d+4 >= g (if d even).
    # Since LUW is bipartite, girth is even.
    # So if g is odd, we need girth >= g+1.
    # If g is even, we need girth >= g.
    
    target_girth = g if g % 2 == 0 else g + 1
    
    # Find smallest d such that girth(d) >= target_girth
    # girth(d) = d+5 if d odd, d+4 if d even.
    d = None
    for cand_d in range(3, 20):
        girth = cand_d + 5 if cand_d % 2 == 1 else cand_d + 4
        if girth >= target_girth:
            d = cand_d
            break
    
    if d is None:
        d = 19
        # If d is too large, LUW might be huge.
        # But for small k, it's fine.
        
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
    
    # Time check during construction
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END

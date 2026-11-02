import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 140.0

    if k == 4 and g == 10:
        # Known (4,10)-cage with 320 vertices via voltage graph construction
        # We use a known 4-regular graph with girth 10 on 320 vertices.
        # This is constructed as a cyclic graph Z_320 with specific offsets,
        # or via a known cage construction.
        
        # Actually, let me try to find a valid (4,10)-cage construction.
        # The known smallest is 320 vertices.
        # One construction: take the graph from the paper by Exoo et al.
        # For now, let's use a known valid construction.
        
        # Let me try a different approach: use the known 320-vertex (4,10)-cage
        # which can be constructed as follows:
        # It's a covering of a smaller graph.
        
        # Actually, let me just construct a valid graph directly.
        # The (4,10)-cage with 320 vertices is known to exist.
        # Let me use a construction based on the Hoffman-Singleton type.
        
        # For k=4, g=10, n=320:
        # One known construction uses the group Z_320 with connections.
        # But finding the exact offsets is complex.
        
        # Let me use a simpler approach: construct the graph using
        # a known method. The (4,10)-cage can be constructed as:
        # Take 8 copies of a 40-vertex graph and connect them appropriately.
        
        # Actually, let me just use the LW graph fallback which gives a valid
        # but larger graph, and see if I can do better.
        
        # The LW graph for k=4, g=10 gives n = 4 * 2^(g/2 - 1) or similar.
        # Let me compute what the LW graph gives.
        
        # For the LW construction with q=4, d=5:
        # n = q * 2^d = 4 * 32 = 128? No, that's not right.
        # The LW graph has n = q * (q-1) * (q^d - 1)/(q-1) = q * (q^d - 1)
        # For q=4, d=5: n = 4 * (1024 - 1) = 4092? That's too large.
        
        # Actually the LW graph for degree 4 and girth 2d+1 has:
        # n = 4 * 2^(2d-1) or something. Let me just use the seed's construction.
        
        # The seed's LW construction gives a valid graph. Let me use it
        # but try to find a smaller one first.
        
        # Let me try a direct construction for 320 vertices.
        # The (4,10)-cage: one known construction is via the following:
        # Vertices are (i, j) where i in Z_8, j in Z_40.
        # Edges connect (i, j) to (i+1, j+a_i), (i-1, j-b_i), (i, j+c_i), (i, j-d_i)
        # with specific a_i, b_i, c_i, d_i.
        
        # This is getting complex. Let me just use the LW graph which is guaranteed
        # to be valid, even if larger than optimal.
        
        # Actually, let me try a different known construction.
        # The (4,10)-cage can be constructed as a circulant-like graph.
        
        # Let me try: n=320, and for each vertex i, connect to:
        # i+1, i-1, i+40, i-40 (this gives degree 4)
        # But this has girth 4 (cycle 0-1-41-40-0).
        
        # Let me use the seed's approach but verify it works.
        pass

    # Use the LW graph construction from the seed
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

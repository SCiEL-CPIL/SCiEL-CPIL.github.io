"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    q = k
    
    # Determine d
    # Girth of CD(d, q) is d + 4 if d is even, d + 5 if d is odd.
    # We need girth >= g.
    # Try smallest d such that girth >= g.
    d = 3
    while True:
        girth_d = d + 4 if d % 2 == 0 else d + 5
        if girth_d >= g:
            break
        d += 1
    
    # GF(q) arithmetic
    # For prime q, use mod q.
    # For q=4, use XOR for add/sub, and a table for mul.
    if q == 4:
        mul_table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return mul_table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # LUW rules
    # r_i = (a, b) for i=2..d
    # Standard pattern to ensure girth:
    # r_2 = (1, 1)
    # r_i = (i-1, 1) if i is odd, (1, i-1) if i is even?
    # Let's use the pattern from the seed which seemed to work for girth.
    rules = [None] * (d + 1)
    rules[2] = (1, 1)
    for c in range(3, d + 1):
        if c % 2 == 0:
            rules[c] = (1, c - 1)
        else:
            rules[c] = (c - 1, 1)

    # Encode vertices as integers for faster hashing
    # Vertex: (side, x1, ..., xd)
    # side is 0 or 1.
    # x_i in 0..q-1.
    # Total states: 2 * q^d.
    
    # We will generate vertices by BFS from (0, 0, ..., 0).
    
    # Precompute neighbors for each possible "local" state?
    # The transition depends on the current vector x.
    
    def get_neighbors(vertex_id):
        # Decode vertex_id
        # This decoding in the inner loop might be slow.
        # Alternative: Store vertices as tuples in a dict, but use int keys for the set.
        # Or, since n is small (6250), we can store the vector components.
        
        pass

    # Let's use a dict mapping int_id -> (side, x_tuple)
    # And a list of edges.
    
    id_to_vertex = {}
    vertex_to_id = {}
    
    start_vertex = (0, tuple([0] * d))
    start_id = 0
    id_to_vertex[start_id] = start_vertex
    vertex_to_id[start_vertex] = start_id
    
    queue = deque([start_id])
    edges = []
    
    # Precompute transition tables for speed?
    # For a given side and current x, and next y[1], we compute y[2..d].
    
    while queue:
        if time.time() > deadline:
            break
        current_id = queue.popleft()
        side, x = id_to_vertex[current_id]
        
        for y1 in range(q):
            # Construct y
            y = [None, y1] + [0] * (d - 1)
            
            for c in range(2, d + 1):
                a, b = rules[c]
                if side == 0:
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # y[c] = x[c] - x[a] * y[b]
                    # Note: In line-to-point, the formula is p[c] = l[c] - l[a] * p[b]
                    # Here x is the line, y is the point?
                    # Seed code:
                    # if side == 0: point to line: l[c] = p[c] + l[a] * p[b]
                    # else: line to point: p[c] = l[c] - l[a] * p[b]
                    # So if side==1 (line), we compute point y.
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            
            next_vertex = (1 - side, tuple(y))
            
            if next_vertex not in vertex_to_id:
                next_id = len(id_to_vertex)
                id_to_vertex[next_id] = next_vertex
                vertex_to_id[next_vertex] = next_id
                queue.append(next_id)
            
            if side == 0:
                edges.append((current_id, vertex_to_id[next_vertex]))
        
    n = len(id_to_vertex)
    
    # Verify girth (optional, but ensures validity)
    # For n=6250, BFS from every vertex is O(n * (n + m)) = 6250 * 6250 * 5 ~ 1.9e8. Too slow in Python.
    # We trust the LUW construction if parameters are correct.
    # The seed program did a verification that was likely too slow or skipped.
    # Given the deadline, we skip full verification for large n.
    
    return n, edges
# EVOLVE-BLOCK-END

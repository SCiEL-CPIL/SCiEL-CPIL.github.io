"""Construct a 4-regular graph of girth 10 using the Lazebnik-Ustimenko-Woldar CD(d, q) construction."""
import time
from collections import deque

def construct(k, g):
    deadline = time.time() + 110.0
    
    # For k=4, g=10, use LUW CD(5, 4)
    # This produces a 4-regular graph with girth >= 10.
    # Size is 2 * 4^(5+1) = 8192.
    # While not optimal (best known is 320), it is a valid explicit construction.
    
    if k != 4 or g != 10:
        # Fallback for other parameters: use LUW with appropriate d
        q = k
        # Girth >= d + 5. We need girth >= g.
        # So d + 5 >= g => d >= g - 5.
        # Also d must be odd for the standard LUW construction to work nicely?
        # The seed code uses: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    else:
        q = 4
        d = 5

    # GF(q) operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Define the rule for LUW graph
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Encode vectors as integers for speed
    # Vector length is d+1 (indices 0..d). Index 0 is unused (None in seed).
    # We store indices 1..d.
    # Value is sum(y[i] * q^(i-1)) for i=1..d.
    
    def encode(vec):
        # vec is list of length d+1, vec[0] is None
        val = 0
        for i in range(1, d + 1):
            val += vec[i] * (q ** (i - 1))
        return val

    def decode(val):
        vec = [None] + [0] * d
        for i in range(1, d + 1):
            vec[i] = val % q
            val //= q
        return vec

    # Neighbours function using integer encoding
    def neighbours(vertex_int):
        # vertex_int encodes the vector x
        x = decode(vertex_int)
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                # point p ~ line l iff l[c] - p[c] = l[a] * p[b]
                # y is line, x is point.
                # If side == 0 (point), we compute line y from point x.
                # y[c] = x[c] + y[a] * x[b]
                # If side == 1 (line), we compute point x from line y.
                # x[c] = y[c] - x[a] * y[b]
                # In our BFS, we start from point (side 0).
                # So we compute y from x.
                y[c] = add(x[c], mul(y[a], x[b]))
            # Yield the line vertex
            yield encode(y)

    # Start from zero point
    start_vec = [None] + [0] * d
    start_int = encode(start_vec)
    
    index = {start_int: 0}
    order = [start_int]
    edges = []
    
    # BFS to build the connected component
    # We need to track the side (point or line) to avoid adding duplicate edges
    # and to know when to add an edge.
    # Side 0: Point, Side 1: Line.
    # Start is Point.
    # Neighbours of Point are Lines.
    # Neighbours of Line are Points.
    
    # We store (int_val, side) in the queue?
    # Actually, the integer encoding is unique for the vector.
    # But a vector can be a point or a line?
    # In LUW, points and lines are disjoint sets.
    # We can distinguish them by side.
    # Let's store side in the index key?
    # Key: (int_val, side)
    
    index = {(start_int, 0): 0}
    order = [(start_int, 0)]
    edges = []
    
    for v_idx, (v_int, v_side) in enumerate(order):
        if time.time() > deadline - 5.0:
            break
            
        # Determine neighbours
        # If v_side == 0 (Point), neighbours are Lines (side 1)
        # If v_side == 1 (Line), neighbours are Points (side 0)
        
        # The neighbour calculation is symmetric in form but different in operation.
        # For Point x -> Line y: y[c] = x[c] + y[a]*x[b]
        # For Line y -> Point x: x[c] = y[c] - x[a]*y[b]
        
        # We need to iterate over the 'free' variable (index 1)
        
        if v_side == 0:
            x = decode(v_int)
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
                n_int = encode(y)
                n_side = 1
                n_key = (n_int, n_side)
                if n_key not in index:
                    index[n_key] = len(order)
                    order.append(n_key)
                # Add edge from Point to Line
                edges.append((v_idx, index[n_key]))
        else:
            y = decode(v_int)
            for first in range(q):
                x = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    x[c] = sub(y[c], mul(x[a], y[b]))
                n_int = encode(x)
                n_side = 0
                n_key = (n_int, n_side)
                if n_key not in index:
                    index[n_key] = len(order)
                    order.append(n_key)
                # Edge already added when the point was processed?
                # In BFS, we add edges when we discover the neighbour.
                # If we start from Point, we add Point->Line.
                # When we process Line, we discover Points.
                # We should only add edges once.
                # Since the graph is bipartite, we can just add edges when side==0 (Point) processes its Line neighbours.
                # Or add when discovering new nodes.
                # The code above adds edge when v_side==0.
                # This covers all edges exactly once (from Point side).
                pass

    n = len(order)
    return n, edges

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct(k, g):
    deadline = time.time() + 140.0
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Precompute field operations as lookup tables for speed
    if q == 4:
        add_tbl = np.zeros((4, 4), dtype=np.int8)
        mul_tbl = np.zeros((4, 4), dtype=np.int8)
        for a in range(4):
            for b in range(4):
                add_tbl[a, b] = a ^ b
                mul_tbl[a, b] = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]][a][b]
        sub_tbl = add_tbl  # In GF(4) char 2, subtraction = addition
    else:
        add_tbl = np.zeros((q, q), dtype=np.int8)
        mul_tbl = np.zeros((q, q), dtype=np.int8)
        sub_tbl = np.zeros((q, q), dtype=np.int8)
        for a in range(q):
            for b in range(q):
                add_tbl[a, b] = (a + b) % q
                sub_tbl[a, b] = (a - b) % q
                mul_tbl[a, b] = (a * b) % q
    
    # Build rule table: rule[c] = (a, b) for c = 2..d
    rule_a = np.zeros(d + 1, dtype=np.int8)
    rule_b = np.zeros(d + 1, dtype=np.int8)
    rule_a[2], rule_b[2] = 1, 1
    rule_a[3], rule_b[3] = 2, 1
    rule_a[4], rule_b[4] = 1, 2
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule_a[c], rule_b[c] = options[r]
    
    # Encode a vertex (side, coords) as a single integer for fast hashing
    # side: 0 or 1, coords: d+1 values in [0, q)
    # Encode: side * q^(d+1) + sum(coords[i] * q^i)
    q_powers = np.array([q**i for i in range(d + 1)], dtype=np.int64)
    
    def encode(side, coords):
        return side * (q ** (d + 1)) + int(np.dot(coords, q_powers))
    
    def decode(code):
        side = code // (q ** (d + 1))
        remainder = code % (q ** (d + 1))
        coords = np.zeros(d + 1, dtype=np.int8)
        for i in range(d + 1):
            coords[i] = remainder % q
            remainder //= q
        return side, coords
    
    # Generate neighbours using vectorized operations
    def get_neighbours(side, coords):
        """Return list of encoded neighbour vertices."""
        neighbours = []
        for first in range(q):
            y = np.zeros(d + 1, dtype=np.int8)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule_a[c], rule_b[c]
                if side == 0:
                    y[c] = add_tbl[coords[c], mul_tbl[y[a], coords[b]]]
                else:
                    y[c] = sub_tbl[coords[c], mul_tbl[coords[a], y[b]]]
            neighbours.append(encode(1 - side, y))
        return neighbours
    
    # BFS from zero vertex
    start_coords = np.zeros(d + 1, dtype=np.int8)
    start_code = encode(0, start_coords)
    
    index = {start_code: 0}
    order = [start_code]
    edges = []
    pos = 0
    
    while pos < len(order):
        if time.time() > deadline:
            break
        code = order[pos]
        pos += 1
        side, coords = decode(code)
        for nb_code in get_neighbours(side, coords):
            if nb_code not in index:
                index[nb_code] = len(order)
                order.append(nb_code)
            if side == 0:
                edges.append((index[code], index[nb_code]))
    
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END

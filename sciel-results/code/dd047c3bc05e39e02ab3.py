"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    or q = 4); the connected component of the zero point is returned.
    
    Optimized: integer-encoded vertices, stored coords, inlined field ops.
    """
    deadline = time.time() + 115
    
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
    
    # Rules: rule[c] = (a, b) meaning y[c] depends on y[a] and x[b] (or x[a] and y[b])
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    rules_a = [0] * (d + 1)
    rules_b = [0] * (d + 1)
    for c in range(2, d + 1):
        rules_a[c], rules_b[c] = rule[c]
    
    r2d = range(2, d + 1)
    
    if q == 4:
        mul_tbl = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
    else:
        mul_tbl = [[(i*j) % q for j in range(q)] for i in range(q)]
    
    # Powers of q for encoding
    qp = [1] * (d + 1)
    for i in range(1, d + 1):
        qp[i] = qp[i-1] * q
    side_base = qp[d] * q  # q^(d+1)
    
    # Field operations (inlined for speed)
    if q == 4:
        def fadd(x, y):
            return x ^ y
        def fsub(x, y):
            return x ^ y
    else:
        def fadd(x, y):
            return (x + y) % q
        def fsub(x, y):
            return (x - y) % q
    
    # BFS with integer-encoded vertices
    # Vertex encoding: code = side * side_base + x[1]*qp[1] + x[2]*qp[2] + ... + x[d]*qp[d]
    # x[0] is always 0, so we store coords as [x[1], ..., x[d]] (length d)
    
    start_coords = [0] * d
    start_code = 0  # side=0, all coords zero
    
    index = {start_code: 0}
    order = [start_code]
    edges = []
    
    # Store side and coords for each vertex to avoid re-decoding
    v_side = [0]
    v_coords = [start_coords]
    
    q_head = 0
    while q_head < len(order):
        if time.time() > deadline:
            break
        
        vi = q_head
        q_head += 1
        vcode = order[vi]
        side = v_side[vi]
        xc = v_coords[vi]
        
        for first in range(q):
            # Compute y[1]..y[d]
            # y[1] = first
            # For c in 2..d:
            #   side 0: y[c] = x[c] + y[a] * x[b]
            #   side 1: y[c] = x[c] - x[a] * y[b]
            # where a < c ensures y[a] is already computed
            
            y = [0] * (d + 1)
            y[1] = first
            
            if side == 0:
                for c in r2d:
                    a = rules_a[c]
                    b = rules_b[c]
                    x_c = xc[c-1]
                    ya = first if a == 1 else y[a]
                    xb = xc[b-1]
                    y[c] = fadd(x_c, mul_tbl[ya][xb])
            else:
                for c in r2d:
                    a = rules_a[c]
                    b = rules_b[c]
                    x_c = xc[c-1]
                    xa = xc[a-1]
                    yb = first if b == 1 else y[b]
                    y[c] = fsub(x_c, mul_tbl[xa][yb])
            
            new_side = 1 - side
            ncode = new_side * side_base
            for i in range(d):
                ncode += y[i+1] * qp[i+1]
            
            if ncode not in index:
                nidx = len(order)
                index[ncode] = nidx
                order.append(ncode)
                v_side.append(new_side)
                v_coords.append(y[1:])
            
            if side == 0:
                edges.append((vi, index[ncode]))
    
    return len(order), edges
# EVOLVE-BLOCK-END

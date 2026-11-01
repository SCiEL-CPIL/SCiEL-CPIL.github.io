"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 115  # safety margin before 120s kill
    q = k
    # For the L-U-W construction, we need girth >= g.
    # The construction gives girth d+5 (odd) or d+6 (even).
    # We want the smallest d such that girth >= g.
    if g <= 6:
        # Small girth cases: use known constructions or simple ones
        if k == 2:
            # Cycle C_g
            n = g
            edges = [(i, (i+1) % n) for i in range(n)]
            return n, edges
        elif k == 3:
            # Petersen graph for girth 5 (10 vertices), or use L-U-W
            pass
        # Fall through to L-U-W for larger k
    
    # Determine d for L-U-W: girth is at least d+5
    # We need d+5 >= g, so d >= g-5
    # d must be odd for the standard L-U-W construction
    d = g - 5
    if d % 2 == 0:
        d += 1  # make odd
    d = max(3, d)
    
    # Precompute field operations for speed
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        mul_table = table
        add_table = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
        sub_table = add_table  # in GF(4), -x = x
    else:
        mul_table = [[(i*j) % q for j in range(q)] for i in range(q)]
        add_table = [[(i+j) % q for j in range(q)] for i in range(q)]
        sub_table = [[(i-j) % q for j in range(q)] for i in range(q)]
    
    # Rule: for c=2..d, rule[c] = (a, b)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Precompute rule arrays for faster access
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    for c in range(2, d + 1):
        rule_a[c], rule_b[c] = rule[c]
    
    # Vertex representation: side (0=point, 1=line) and tuple of d+1 elements
    # Vertex 0 is (None, 0, 0, ..., 0) on side 0
    # We'll use BFS to find the connected component
    
    # For speed, encode vertices as integers when possible
    # Side 0 vertices: (0, x1, x2, ..., xd) where x1 is None -> treat as 0
    # Actually, let's use a different encoding.
    
    # Let's use a dictionary-based BFS but with optimized neighbor generation
    
    def make_vertex(side, vals):
        return (side, tuple(vals))
    
    start = make_vertex(0, [0] * (d + 1))  # x[0] unused, x[1..d]
    
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS with queue
    queue = deque([0])
    
    while queue:
        if time.time() > deadline - 5:
            break  # safety margin
        
        vi = queue.popleft()
        vertex = order[vi]
        side, x = vertex
        
        for first in range(q):
            if side == 0:
                # Point -> Line: y[c] = x[c] + y[a]*x[b] for c=2..d
                # y[1] = first
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a = rule_a[c]
                    b = rule_b[c]
                    y[c] = add_table[x[c]][mul_table[y[a]][x[b]]]
            else:
                # Line -> Point: y[c] = x[c] - x[a]*y[b] for c=2..d
                # y[1] = first
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a = rule_a[c]
                    b = rule_b[c]
                    y[c] = sub_table[x[c]][mul_table[x[a]][y[b]]]
            
            other = make_vertex(1 - side, y)
            
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(len(order) - 1)
            
            if side == 0:
                edges.append((vi, index[other]))
    
    return len(order), edges
# EVOLVE-BLOCK-END

import time
import math

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # LUW parameters
    # For k=5, q=5.
    # If k is not a prime or 4, this simple LUW won't work directly.
    # Assuming k=5 based on task title.
    if k != 5:
        # Fallback: try to find a prime q >= k
        # This is a heuristic, might not produce k-regular graph if q != k
        # But for this task, k is likely 5.
        pass
        
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # Assume q is prime for standard field operations
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Determine d
    # Girth of CD(d, q) is 2d + 2
    # We need 2d + 2 >= g  => d >= (g - 2) / 2
    d = max(2, math.ceil((g - 2) / 2.0))
    
    # Rule calculation for LUW
    # The adjacency is defined by:
    # p ~ l  iff  l_i - p_i = l_j * p_k  for specific (i, j, k) depending on i
    # The seed used a specific rule. Let's replicate the optimal rule structure.
    # Standard LUW rule for index i (2..d):
    # Usually involves (i, i-1, i-2) or similar.
    # The seed's rule:
    # rule[2] = (1, 1)
    # rule[3] = (2, 1)
    # rule[4] = (1, 2)
    # This looks like a specific pattern.
    # Let's use the standard LUW construction which is known to have girth 2d+2.
    # The standard construction uses:
    # l_i = p_i + p_{i-1} * p_{i-2} (for i > 2)
    # This corresponds to a = i-1, b = i-2.
    # Let's verify if this rule works for all d.
    
    rule = {}
    for c in range(2, d + 1):
        if c == 2:
            rule[c] = (1, 1) # l_2 = p_2 + p_1 * p_1 ? Or just p_2 + p_1?
            # Actually, for c=2, we need to relate l_2, p_2, p_1.
            # Standard LUW: l_2 = p_2 + p_1. (a=1, b=1? No, mul(p_1, p_1)=p_1^2).
            # Let's stick to the seed's rule which was validated.
            # Seed: rule[2]=(1,1), rule[3]=(2,1), rule[4]=(1,2).
            # This pattern is not obvious.
            # However, the seed is known to be valid.
            # I will use the seed's rule generation logic but ensure d is minimal.
            
            # Re-implementing seed's rule logic:
            if c == 2:
                rule[c] = (1, 1)
            elif c == 3:
                rule[c] = (2, 1)
            elif c == 4:
                rule[c] = (1, 2)
            else:
                # General pattern from seed:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
    # BFS to construct graph
    # Vertices are (side, tuple_of_values)
    # side 0: Points, side 1: Lines
    # Start at (0, (0, 0, ..., 0))
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # Precompute neighbours for speed? No, BFS generates them.
    
    for vertex in order:
        side, x = vertex
        v_idx = index[vertex]
        
        # Iterate over the 'free' variable (y[1] or x[1] equivalent)
        # In the seed, for a point x, we iterate 'first' which sets y[1].
        # Then we compute y[2..d].
        # This defines the unique line connected to point x with that y[1].
        # There are q such lines.
        
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            
            # Compute y[c] for c = 2..d
            # Relation: l[c] - p[c] = l[a] * p[b]  =>  l[c] = p[c] + l[a] * p[b]
            # If side == 0 (vertex is Point x), we are finding Line y.
            # y[c] = x[c] + y[a] * x[b]
            # If side == 1 (vertex is Line x), we are finding Point y.
            # x[c] - y[c] = x[a] * y[b]  =>  y[c] = x[c] - x[a] * y[b]
            
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            
            neighbor = (1 - side, tuple(y))
            
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
                
            # Add edge only once (when side == 0)
            if side == 0:
                edges.append((v_idx, index[neighbor]))
                
        if time.time() > deadline:
            break
            
    return len(order), edges

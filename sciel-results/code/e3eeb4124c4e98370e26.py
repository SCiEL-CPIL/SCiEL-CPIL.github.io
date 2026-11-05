import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin

    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    # Degree is q. We set q = k.
    # Girth is at least 2*d + 2 (for bipartite L-W graphs).
    # We need girth >= g.
    # So 2*d + 2 >= g => d >= (g - 2) / 2.
    # d must be an integer >= 1.
    
    if k < 3:
        # Fallback for small k, though task is k=5
        return 0, []

    q = k
    
    # Calculate minimal d
    # d = ceil((g - 2) / 2)
    d = max(1, (g - 1) // 2)
    
    # If q is 4, use specific field arithmetic
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Rule for L-W construction
    # rule[c] = (a, b) such that y[c] = x[c] +/- y[a]*x[b]
    # The specific rule depends on c and d.
    # Standard rule:
    # For c=2..d:
    # If c is even: y[c] = x[c] + y[c-1]*x[c-1] ? No.
    # The rule in the seed code is:
    # rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    # where r = (c - 5) % 4, c0 = c - r.
    # This seems to be a specific implementation of the L-W recurrence.
    # We will reuse this rule logic as it is known to work for the seed.
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1. x[0] is None for points?
        # In seed: start = (0, (None,) + (0,) * d)
        # So x has d+1 elements. x[0] is index 0.
        # The loop in seed: for first in range(q): y = [None, first] + [0]*(d-1)
        # So y[0] is None, y[1] is first, y[2..d] are computed.
        
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                # Indices in x and y correspond to 1..d
                # x[a] and x[b] are used.
                # Note: x[0] is None, so a,b must be >= 1.
                # In the rule, a,b are derived from c.
                # For c=2, rule=(1,1). a=1, b=1. x[1] is valid.
                
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    # BFS to find connected component
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        if time.time() > deadline:
            break
        vertex = order[i]
        i += 1
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge only once (from point to line)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

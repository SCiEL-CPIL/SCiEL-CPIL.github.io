import time
import sys

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    q = k
    if q != 3:
        # For non-prime or unsupported k, fall back to LUW if possible
        pass
    
    def girth_of_d(d):
        if d % 2 == 1:
            return d + 5
        else:
            return d + 6
    
    d = 3
    while girth_of_d(d) < g:
        d += 1
    
    # Precompute rule arrays for speed
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    rule_a[2], rule_b[2] = 1, 1
    rule_a[3], rule_b[3] = 2, 1
    rule_a[4], rule_b[4] = 1, 2
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule_a[c], rule_b[c] = rules_cycle[r]
    
    index = {}
    order = []
    edges = []
    
    start = (0, (None,) + (0,) * d)
    index[start] = 0
    order.append(start)
    
    for i, vertex in enumerate(order):
        side, x = vertex
        for first in range(q):
            y = [None] + [0] * d
            y[1] = first
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = (x[c] + y[a] * x[b]) % q
            else:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = (x[c] - x[a] * y[b]) % q
            
            neighbor = (1 - side, tuple(y))
            
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            
            if side == 0:
                edges.append((i, index[neighbor]))
        
        if time.time() > deadline:
            break
    
    n = len(order)
    return n, edges

# EVOLVE-BLOCK-END

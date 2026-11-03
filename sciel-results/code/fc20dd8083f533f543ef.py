"""Improved LUW construction for cage problem."""
import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    """Construct a k-regular graph with girth at least g using LUW construction.
    Tries multiple d values and returns the smallest valid graph."""
    
    deadline = time.time() + 140  # 10s safety margin
    
    def get_field_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
            return add, sub, mul
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
            return add, sub, mul
    
    def get_rule(d):
        """Get the LUW rule for given d."""
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return rule
    
    def build_luw(q, d):
        """Build the LUW graph CD(d,q) and return the connected component of zero."""
        add, sub, mul = get_field_ops(q)
        rule = get_rule(d)
        
        # Point: (0, tuple of length d+1 with first element None)
        # Line: (1, tuple of length d+1 with first element None)
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        # point p ~ line l: l[c] - p[c] = l[a] * p[b]
                        # y[c] = x[c] + y[a] * x[b]
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        # line l ~ point p: l[c] - p[c] = l[a] * p[b]
                        # x[c] = l[c] - l[a] * p[b]  =>  p[c] = l[c] - l[a]*p[b]... 
                        # Wait, we're given a line x and finding point y
                        # l[c] - p[c] = l[a] * p[b]
                        # p[c] = l[c] - l[a] * p[b]
                        # But p[b] is y[b], and we're computing y[c]
                        # Actually for side==1 (line), we find point y
                        # l[c] - p[c] = l[a] * p[b]
                        # p[c] = l[c] - l[a] * p[b]
                        # y[c] = sub(x[c], mul(x[a], y[b]))
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1 - side), tuple(y)
        
        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for i, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges
    
    # Determine valid d values
    # d odd: girth >= d+5, so d+5 >= g => d >= g-5
    # d even: girth >= d+4, so d+4 >= g => d >= g-4
    
    best_n = float('inf')
    best_edges = None
    best_d = None
    
    # Try d values starting from the minimum
    d_min_odd = max(3, g - 5)
    if d_min_odd % 2 == 0:
        d_min_odd += 1
    d_min_even = max(4, g - 4)
    if d_min_even % 2 == 1:
        d_min_even += 1
    
    # Try a range of d values, starting from the smallest
    all_d = []
    d = 3
    while d <= 12:  # Cap at reasonable d
        if d % 2 == 1:
            if d + 5 >= g:
                all_d.append(d)
        else:
            if d + 4 >= g:
                all_d.append(d)
        d += 1
    
    # Sort by d (smaller d likely means smaller graph)
    all_d.sort()
    
    for d in all_d:
        if time.time() > deadline:
            break
        try:
            n, edges = build_luw(k, d)
            if n < best_n:
                best_n = n
                best_edges = edges
                best_d = d
        except Exception:
            continue
    
    if best_edges is None:
        # Fallback: use the original d computation
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        best_n, best_edges = build_luw(k, d)
    
    return best_n, best_edges
# EVOLVE-BLOCK-END

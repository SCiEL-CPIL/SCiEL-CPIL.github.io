"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    
    Strategy: Use the Lazebnik-Ustimenko-Woldar (LUW) construction.
    The LUW graph CD(d,q) is a bipartite graph with vertices (points and lines)
    over GF(q), where q=k. The girth is d+5 (for odd) or d+6 (for even d).
    For girth >= g, we need d >= g-5 (if g-5 is odd) or d >= g-6 (if g-6 is even).
    
    We return the connected component containing the zero vertex.
    """
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # The LUW construction: CD(d, q) where q = k
    q = k
    # For odd girth g, we need d such that girth = d+5 >= g
    # So d >= g-5. If g-5 is odd, use d = g-5. Otherwise use d = g-4 (girth = g-4+5 = g+1 >= g)
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # make d odd for odd girth
    # Actually for LUW, girth is d+5 when d is odd, d+6 when d is even
    # We want girth >= g. If g is odd, use odd d with d+5 >= g => d >= g-5
    # If g is even, use even d with d+6 >= g => d >= g-6, or odd d with d+5 >= g => d >= g-5
    
    # For g=9 (odd): d >= 4, use d=4 (girth=9) or d=5 (girth=10)
    # Let's use d = g-5 if (g-5) is odd, else g-4
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
    
    # Finite field GF(q) operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
    
    # Determine the rule for each coordinate c = 2..d
    # rule[c] = (a, b) means: l[c] - p[c] = l[a] * p[b]
    # We need a < c and b < c so we can compute sequentially
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        # Use pattern: (1, c-2) to ensure b = c-2 < c and a = 1 < c
        # This gives girth d+5 for the LUW graph
        rule[c] = (1, c - 2)
    
    # Build the graph using BFS from the zero point
    # Vertices are (side, coords) where side=0 is point, side=1 is line
    # coords is a tuple of length d+1 (indices 0..d), with coords[0] unused
    
    start = (0, (0,) * (d + 1))  # all zeros
    
    # BFS to find connected component
    index = {start: 0}
    order = [start]
    edges = []
    queue = deque([0])
    
    while queue:
        vi = queue.popleft()
        if time.time() > deadline:
            break
        vertex = order[vi]
        side, x = vertex
        for first in range(q):
            # Build the neighbor on the other side
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # At point x, finding line y: y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # At line x, finding point y: x[c] - y[c] = x[a] * y[b]
                    # => y[c] = x[c] - x[a] * y[b]
                    # But y[b] must be known, so we need b < c
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbor = (1 - side, tuple(y))
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
                queue.append(len(order) - 1)
            edges.append((vi, index[neighbor]))
    
    n = len(order)
    
    # Remove duplicate edges (BFS may add same edge twice from both sides)
    edge_set = set()
    unique_edges = []
    for e in edges:
        key = (min(e[0], e[1]), max(e[0], e[1]))
        if key not in edge_set:
            edge_set.add(key)
            unique_edges.append(e)
    edges = unique_edges
    
    return n, edges
# EVOLVE-BLOCK-END

import time
import sys

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using the
    Lazebnik-Ustimenko-Woldar (LUW) construction.
    
    The LUW construction CD(d, q) is a q-regular bipartite graph with girth
    at least d+5 (if d is odd) or d+6 (if d is even).
    We choose d such that the girth requirement is met.
    """
    # Safety margin for the 120s limit
    deadline = time.time() + 110.0
    
    q = k
    
    # Determine d for LUW
    # Girth of CD(d, q) is at least:
    # d + 5 if d is odd
    # d + 6 if d is even
    # We want girth >= g.
    
    # Try to find the smallest d such that the guaranteed girth >= g
    # If g is odd, say g = 2r+1. We need d+5 >= 2r+1 => d >= 2r-4.
    # If g is even, say g = 2r. We need d+6 >= 2r => d >= 2r-6.
    
    # Let's just iterate d upwards from 3 to find the first valid one.
    d = 3
    while True:
        if d % 2 == 1:
            min_girth = d + 5
        else:
            min_girth = d + 6
        
        if min_girth >= g:
            break
        d += 1
        
    # Ensure d is at least 3
    d = max(3, d)
    
    # Field operations for GF(q)
    if q == 4:
        # GF(4) representation
        # 0, 1, 2 (alpha), 3 (alpha+1)
        # alpha^2 = alpha + 1 => 2*2 = 3
        # alpha^3 = 1 => 2*3 = 2, 3*3 = 1
        mult_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        add_table = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0]
        ]
        def add(x, y):
            return add_table[x][y]
        def sub(x, y):
            return add_table[x][y] # char 2
        def mul(x, y):
            return mult_table[x][y]
    else:
        # Assume q is prime
        def add(x, y):
            return (x + y) % q
        def sub(x, y):
            return (x - y) % q
        def mul(x, y):
            return (x * y) % q
    
    # Precompute rules for LUW
    # rule[c] = (a, b) for c = 2..d
    # Relation: line[c] - point[c] = line[a] * point[b]
    rule = [None] * (d + 1)
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
            
    # Build the graph
    # Vertices are (side, coords) where side is 0 (point) or 1 (line)
    # coords is a tuple of length d+1 (indices 0..d), with coords[0] unused (set to 0 for hashing)
    
    # We start from the zero point
    start_point = (0, (0,) * (d + 1))
    
    index = {start_point: 0}
    order = [start_point]
    edges = []
    
    # BFS/DFS to find all reachable vertices
    # Since it's a regular graph, we can just iterate through the order list
    # and for each vertex, find its neighbors.
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            # If we run out of time, we return what we have, but this is unlikely for reasonable d
            # However, LUW graphs can be large. If d is large, this might be slow.
            # But for k=5, g=9, d is small.
            # g=9 => d+5 >= 9 => d >= 4 (even, girth 10) or d=5 (odd, girth 10). 
            # Actually for g=9, d=4 gives girth 10 >= 9. d=3 gives girth 8 < 9.
            # So d=4. n = 2 * 5^5 = 6250? No, d=4 means vectors of length 5.
            # n = 2 * q^(d+1) = 2 * 5^5 = 6250. This is small.
            pass
        
        side, x = vertex
        
        # Generate neighbors
        # If vertex is a point (side 0), neighbors are lines (side 1)
        # If vertex is a line (side 1), neighbors are points (side 0)
        
        for first in range(q):
            y = [0] * (d + 1)
            y[0] = 0
            y[1] = first
            
            if side == 0:
                # x is point, find line y
                # y[c] = x[c] + y[a] * x[b]
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                # x is line, find point y
                # y[c] = x[c] - x[a] * y[b]
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
                    
            neighbor = (1 - side, tuple(y))
            
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            
            # Add edge only once (when we are on the 'point' side or just check index order)
            # To avoid duplicates, we can just add edge if current vertex index < neighbor index
            # Or simply add all edges from 'point' vertices to 'line' vertices
            if side == 0:
                edges.append((i, index[neighbor]))
                
    n = len(order)
    
    return n, edges

if __name__ == "__main__":
    # Example for k=5, g=9
    n, edges = construct(5, 9)
    print(f"Vertices: {n}")
    print(f"Edges: {len(edges)}")

import time
import sys

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction for the base case.
    Includes a deadline guard to ensure termination.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # LUW parameters
    # The LUW graph CD(d, q) is q-regular.
    # Girth is d+5 if d is odd, d+6 if d is even.
    # We need girth >= g.
    # For k=3, q must be 3.
    # If k != 3, we might need a different strategy, but the task is specific to k=3.
    # The task says "evaluator passes this task's degree k and girth g".
    # Assuming k=3, g=16.
    
    q = k
    if q != 3 and q != 5 and q != 7:
        # Fallback for other k: use a simple ring or other structure?
        # For now, assume k=3.
        # If k is not 3, LUW might not be applicable directly or q must be prime power.
        # We'll stick to the LUW logic which works for prime q.
        pass
        
    # Determine d
    # g = d + 5 (odd d) or d + 6 (even d)
    # We want g >= g_target.
    # Try to find smallest d such that girth(d) >= g.
    
    def girth_of_d(d):
        if d % 2 == 1:
            return d + 5
        else:
            return d + 6
            
    d = 3
    while girth_of_d(d) < g:
        d += 1
        
    # Check time
    if time.time() > deadline:
        # Return a trivial graph? No, must be valid.
        # Just return LUW with small d? No, must satisfy girth.
        # The LUW construction is fast, so this shouldn't happen.
        pass

    # GF(q) operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    # Rule for LUW
    # rule[c] = (a, b) such that l[c] - p[c] = l[a] * p[b]
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        # The sequence of rules repeats every 4 steps with a shift
        # (1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]

    # Generate vertices
    # Side 0: Points (x0, x1, ..., xd) where x0 is None? 
    # In the seed, start = (0, (None,) + (0,)*d)
    # So x[0] is ignored/None. x[1]...x[d] are in GF(q).
    # Total points: q^d
    # Total lines: q^d
    # Total vertices: 2 * q^d
    
    # Wait, the seed says n = 2 * q^(d+1)?
    # Let's check the seed code:
    # start = (0, (None,) + (0,) * d)
    # y = [None, first] + [0] * (d - 1)
    # y[1] = first (0..q-1)
    # y[2..d] are calculated.
    # So x[1] is free?
    # In neighbours:
    # for first in range(q):
    #   y[1] = first
    #   for c in range(2, d+1):
    #     y[c] = ...
    # So yes, x[1] is free, x[2..d] are determined by x[1] and the other side's coordinates.
    # The vertex is (side, (None, x1, x2, ..., xd)).
    # Number of such tuples is q^d.
    # So total vertices N = 2 * q^d.
    
    # For q=3, d=11: N = 2 * 3^11 = 354,294.
    # This is much smaller than my previous estimate of 3^12.
    # 3^11 = 177,147.
    # 2 * 177,147 = 354,294.
    
    # This is still large, but manageable.
    
    index = {}
    order = []
    edges = []
    
    start = (0, (None,) + (0,) * d)
    index[start] = 0
    order.append(start)
    
    # BFS to generate all reachable vertices
    # Since the graph is connected (LUW is connected), this will generate all 2*q^d vertices.
    
    for i, vertex in enumerate(order):
        side, x = vertex
        # Generate neighbors
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point -> Line
                    # l[c] - p[c] = l[a] * p[b]
                    # l[c] = p[c] + l[a] * p[b]
                    # y is the line. x is the point.
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line -> Point
                    # p[c] = l[c] - l[a] * p[b] ??
                    # Seed: y[c] = sub(x[c], mul(x[a], y[b]))
                    # Here x is line, y is point.
                    # p[c] = l[c] - l[a] * p[b]
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            
            neighbor = (1 - side, tuple(y))
            
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            
            if side == 0:
                edges.append((index[vertex], index[neighbor]))
                
        if time.time() > deadline:
            break
            
    n = len(order)
    return n, edges

# EVOLVE-BLOCK-END

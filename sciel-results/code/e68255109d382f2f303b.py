import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k == 5 and g == 9:
        # Use a construction based on UT(3,5) with optimized generators
        result = _ut3_construct(5, g)
        if result is not None and result[0] < 1250:
            return result
        return _luw_construct(k, g)
    
    return _luw_construct(k, g)

def _ut3_construct(q, g):
    """
    Construct a (q+1)-regular graph using UT(3,q) with q=5.
    UT(3,5) has 125 elements. With 5 generators, we get a 5-regular graph.
    """
    if q != 5:
        return None
    
    # UT(3,5): matrices [[1,a,c],[0,1,b],[0,0,1]] with a,b,c in F_5
    # Generators: x = [[1,1,0],[0,1,0],[0,0,1]], y = [[1,0,0],[0,1,1],[0,0,1]], 
    #              z = [[1,0,1],[0,1,0],[0,0,1]], and two more
    
    # Element represented as (a, b, c)
    # Multiplication: (a1,b1,c1) * (a2,b2,c2) = (a1+a2, b1+b2, c1+c2+a1*b2)
    
    def mul(e1, e2):
        a1, b1, c1 = e1
        a2, b2, c2 = e2
        return ((a1+a2) % q, (b1+b2) % q, (c1+c2+a1*b2) % q)
    
    def inv(e):
        a, b, c = e
        return ((-a) % q, (-b) % q, (-c + a*b) % q)
    
    # 5 generators (and their inverses for the graph)
    # For a 5-regular graph, we need 5 generators where each generator's inverse is also a generator
    # OR use 10 generators (5 pairs of inverses) for a bipartite construction
    
    # Let's use 5 generators where the set is closed under inversion
    # g1 = (1,0,0), g1^-1 = (4,0,0)
    # g2 = (0,1,0), g2^-1 = (0,4,0)  
    # g3 = (0,0,1), g3^-1 = (0,0,4)
    # g4 = (1,1,0), g4^-1 = (4,4,0)
    # g5 = (1,0,1), g5^-1 = (4,0,4)
    
    generators = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1)
    ]
    
    # Check if this set is closed under inversion
    # g1^-1 = (4,0,0) - not in generators!
    # So this won't give a regular graph directly
    
    # Better: use a set S where S = S^-1
    # Let's use: S = {(1,0,0), (4,0,0), (0,1,0), (0,4,0), (0,0,1), (0,0,4)} - that's 6
    # Or: S = {(1,0,0), (0,1,0), (0,0,1), (1,1,1), (4,4,4)} - check inversion
    
    # Actually, for a k-regular graph, we need |S| = k where S is a set of group elements
    # and the graph has edges (g, g*s) for s in S.
    # For this to be undirected, we need S = S^-1.
    
    # Let me use: S = {(1,0,0), (4,0,0), (0,1,0), (0,4,0), (1,1,0)} 
    # But (1,1,0)^-1 = (4,4,0) which is not in S.
    
    # This is getting complex. Let me just return None and use LUW.
    return None

def _luw_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

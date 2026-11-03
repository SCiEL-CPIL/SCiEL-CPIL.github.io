"""
Deterministic construction of a 4-regular graph with girth >= 10 using a
"perfect code" / non-backtracking walk neighborhood approach.

Strategy:
- Use the known explicit construction from the Lazebnik-Ustimenko-Woldar family,
  but optimize the parameter selection to minimize the number of vertices.
- The LUW construction CD(d, q) with q=4 gives a 4-regular bipartite graph.
- For girth g=10, we need d such that the girth is at least d+5 (for odd d) 
  or d+4 (for even d). We want the smallest n.
- With q=4, d=5 gives girth >= 10 (since d+5=10 for odd d=5).
- The number of vertices is 2 * q^(d+1) = 2 * 4^6 = 8192? No, let me reconsider.

Actually, the LUW graph CD(d, q) has 2*q^(d+1) vertices for the full graph,
but the connected component of zero may be smaller. Let me just use the 
seed program's approach but verify it's correct and add a deadline guard.

For k=4, g=10:
- q = 4
- d = g - 5 = 5 (since g-5=5 is odd, use d=5)
- The graph has 2 * 4^(d+1) = 2 * 4^6 = 8192 vertices? That seems too many.

Wait, let me re-examine. The LUW construction: points and lines are vectors 
over GF(q) of length d+1 (indices 1..d). The zero point is (None, 0, ..., 0).
The neighbor relation is defined by the rule. The connected component of zero 
has at most 2 * (q-1)^d * ... no, it's bounded by the Moore bound times 2.

The Moore bound for (4,10) is 1 + 4*sum_{i=0}^{4} 3^i = 1 + 4*(1+3+9+27+81) = 1+4*121 = 485.
But the best known is 320, and lower bound is 243.

The LUW construction with q=4, d=5: the number of vertices in the component 
of zero is at most 2 * (1 + 3 + 3^2 + 3^3 + 3^4 + 3^5) = 2*(1+3+9+27+81+243) = 2*364 = 728.
That's more than 320.

Let me just use the seed program as-is since it produces a valid graph, and add 
a deadline guard. The score is 1.0 which means it matches the seed. To improve, 
I need to find a smaller graph.

Alternative: Use a known explicit 4-regular girth 10 graph with fewer vertices.
The Hoffman-Singleton graph is (3,5) with 50 vertices. For (4,10), there's no 
simple explicit construction I know of with 320 vertices that I can code from scratch.

Let me stick with the LUW construction but verify it works and add safeguards.
"""
import time

def construct(k, g):
    deadline = time.time() + 110  # 110s safety margin
    
    q = k
    # For girth g, we need d such that d+5 >= g (if d odd) or d+4 >= g (if d even)
    # Try to minimize d
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # make d odd so girth = d+5 >= g
    
    # For q=4, use GF(4) with XOR addition and multiplication table
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y):
            return x ^ y
        def sub(x, y):
            return x ^ y
        def mul(x, y):
            return table[x][y]
    else:
        def add(x, y):
            return (x + y) % q
        def sub(x, y):
            return (x - y) % q
        def mul(x, y):
            return x * y % q
    
    # Build the rule for c=2..d
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
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
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

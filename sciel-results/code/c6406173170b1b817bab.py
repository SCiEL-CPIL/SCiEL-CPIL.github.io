import time

def construct(k, g):
    """
    Construct a k-regular graph with girth >= g using the Lazebnik-Ustimenko-Woldar (LUW) construction.
    For the specific task k=5, g=9, we use q=5 (to match degree) and d=5 (to ensure girth >= 10).
    The resulting graph has 2 * 5^5 = 6250 vertices.
    """
    # The LUW graph CD(d, q) is q-regular and has girth >= d+5 for odd d.
    # We need degree k, so we set q = k.
    # We need girth >= g.
    # If d is odd, girth >= d+5. We need d+5 >= g => d >= g-5.
    # If d is even, girth >= d+4. We need d+4 >= g => d >= g-4.
    # We want the smallest d to minimize vertices 2*q^d.
    
    q = k
    if (g - 5) % 2 == 0:
        # g-5 is even, so g is odd? 
        # If g=9, g-5=4 (even). d must be odd to use the d+5 bound?
        # Actually, the bound is:
        # d odd: girth >= d+5
        # d even: girth >= d+4
        # We want the smallest d such that the bound holds.
        # Try odd d: d >= g-5. Smallest odd >= g-5.
        # Try even d: d >= g-4. Smallest even >= g-4.
        # Compare 2*q^d for both.
        
        d_odd = g - 5
        if d_odd % 2 == 0:
            d_odd += 1
        if d_odd < 3: d_odd = 3 # min d for LUW is usually 3 or 4
        
        d_even = g - 4
        if d_even % 2 == 1:
            d_even += 1
        if d_even < 4: d_even = 4
        
        # Choose the one with smaller 2*q^d
        if q ** d_odd <= q ** d_even:
            d = d_odd
        else:
            d = d_even
    else:
        # g-5 is odd, so g is even?
        # e.g. g=8. g-5=3 (odd).
        # d odd: d >= 3.
        # d even: d >= 4.
        d_odd = g - 5
        if d_odd % 2 == 0:
            d_odd += 1
        if d_odd < 3: d_odd = 3
        
        d_even = g - 4
        if d_even % 2 == 1:
            d_even += 1
        if d_even < 4: d_even = 4
        
        if q ** d_odd <= q ** d_even:
            d = d_odd
        else:
            d = d_even

    # Build the LUW graph
    # q is prime (5), so we use standard mod arithmetic
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return (x * y) % q

    # Define the rule for LUW
    # rule[c] = (a, b) such that y[c] = y[a] * x[b] (or similar)
    # Standard LUW rule:
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        # The sequence of rules repeats every 4
        # c=5: r=0, c0=5. rule[5] = (1, 3) ? 
        # Let's use the explicit formula from the seed/inspiration:
        # [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1: (None, x1, x2, ..., xd)
        # Wait, the seed uses (None, first) + [0]*(d-1).
        # So x has indices 0..d. x[0] is None. x[1]..x[d] are values.
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to Line
                    # l[c] - p[c] = l[a] * p[b]
                    # y[c] = p[c] + l[a] * p[b] ?
                    # Seed: y[c] = add(x[c], mul(y[a], x[b]))
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line to Point
                    # p[c] = l[c] - l[a] * p[b]
                    # Seed: y[c] = sub(x[c], mul(x[a], y[b]))
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge only once (from point side)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

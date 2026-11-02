import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    # q-regular bipartite graph
    # Girth is at least 2 * floor((d+1)/2) + 4
    # We need girth >= g
    # So 2 * floor((d+1)/2) + 4 >= g
    # floor((d+1)/2) >= (g-4)/2
    # Let m = floor((d+1)/2). We need m >= ceil((g-4)/2)
    # d = 2m - 1 or 2m - 2
    
    q = k
    if q not in [2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        # If k is not prime or 4, LUW is not directly applicable in this simple form.
        # But the task implies k=5.
        pass

    # Determine minimal d
    # m = ceil((g-4)/2)
    m = max(1, (g - 4 + 1) // 2) # ceil((g-4)/2)
    # d can be 2m-1 or 2m-2
    # Smallest d is 2m-2?
    # If d=2m-2, floor((d+1)/2) = floor((2m-1)/2) = m-1.
    # Girth = 2(m-1)+4 = 2m+2.
    # We need 2m+2 >= g?
    # If g=9, m=ceil(5/2)=3.
    # d=4. Girth = 2(2)+4 = 8. Not enough.
    # So we need m such that 2m+2 >= g?
    # No, the formula is 2*floor((d+1)/2) + 4.
    # Let d=2m-1. floor((2m)/2)=m. Girth = 2m+4.
    # Let d=2m-2. floor((2m-1)/2)=m-1. Girth = 2(m-1)+4 = 2m+2.
    
    # We want Girth >= g.
    # Case 1: d=2m-2. 2m+2 >= g => m >= (g-2)/2.
    # Case 2: d=2m-1. 2m+4 >= g => m >= (g-4)/2.
    
    # We want minimal d.
    # Try small d values.
    d = 1
    while True:
        girth = 2 * ((d + 1) // 2) + 4
        if girth >= g:
            break
        d += 1
        if time.time() > deadline:
            break

    # If q == 4, use GF(4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Rule for LUW graph
    # y[c] = x[c] + y[a] * x[b] (for points to lines)
    # y[c] = x[c] - x[a] * y[b] (for lines to points)
    # Indices: 0..d
    # c goes from 2 to d
    # a, b depend on c
    
    rule = {}
    # Standard LUW rules
    # For c=2: a=1, b=1
    # For c=3: a=2, b=1
    # For c=4: a=1, b=2
    # For c>4: pattern repeats?
    # The seed code used a specific pattern. I will use the seed's pattern.
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
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
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END

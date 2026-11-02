import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    q = k
    # For girth g, we need d such that girth >= g.
    # LUW girth is 2d+2 for odd d? Or d+5?
    # The seed code uses: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=16: g-5=11 (odd), so d=11.
    # Girth of CD(11, 3) is at least 16.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # Precompute rules
    rule = {}
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        # The list of rules for c
        # r=0: (1, c0-2)
        # r=1: (c0-1, 1)
        # r=2: (c0, 1)
        # r=3: (1, c0+1)
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # To speed up, we can use integer representation for vertices
    # Vertex is (side, x0, ..., xd)
    # x0 is ignored in neighbor calculation?
    # In seed: y = [None, first] + [0]*(d-1)
    # y[0] is None. y[1] is 'first'.
    # So the vector is (x0, x1, ..., xd) where x0 is effectively the "first" variable?
    # The seed code uses `x[c]` for c=2..d.
    # And `y[a]` where `a` can be 1.
    # So `x` needs index 1..d.
    # `x[0]` is not used in the loop `for c in range(2, d+1)` except `x[c]`.
    # Wait, `x` is the vertex tuple.
    # `side, x = vertex`.
    # `x` has length d+1.
    # `x[0]` is the first element.
    # In `neighbours`, `y` is initialized with `first` at index 1.
    # So `x[1]` is the variable being iterated?
    # No, `first` is the loop variable.
    # So `x[1]` is not used to generate neighbors?
    # Let's check `y[c] = add(x[c], mul(y[a], x[b]))`.
    # `x` is the current vertex.
    # So `x[1]` is used if `a` or `b` is 1?
    # `a, b = rule[c]`.
    # For c=2, rule is (1,1). So `a=1, b=1`.
    # `y[2] = add(x[2], mul(y[1], x[1]))`.
    # So `x[1]` is used.
    # `x[0]` is never used?
    # The start vertex is `(0, (None,) + (0,)*d)`.
    # So `x[0]` is None.
    # It seems `x[0]` is a dummy.
    
    # We can represent the vertex as an integer to speed up hashing and storage.
    # Since q=3, we can pack the tuple into an integer.
    # d=11. 12 elements (0..11).
    # 12 * 2 bits = 24 bits. Fits in int.
    
    def encode(side, x_tuple):
        # x_tuple is length d+1. x[0] is ignored/dummy.
        # We pack x[1]..x[d] and side.
        val = side
        for i in range(1, d+1):
            val = val * q + x_tuple[i]
        return val

    def decode(val):
        side = val % 2
        val //= 2
        x = [0] * (d + 1)
        for i in range(d, 0, -1):
            x[i] = val % q
            val //= q
        # x[0] is dummy
        return side, tuple(x)

    # Precompute neighbor offsets?
    # The neighbor calculation is:
    # y[1] = first
    # for c in 2..d:
    #   a, b = rule[c]
    #   y[c] = x[c] + y[a]*x[b] (mod q)  (if side 0)
    #   y[c] = x[c] - x[a]*y[b] (mod q)  (if side 1)
    
    # We can precompute the transformation for each `first` (0..q-1)
    # But it depends on `x`.
    
    # BFS
    start_val = encode(0, (0,) * (d + 1))
    index = {start_val: 0}
    order = [start_val]
    edges = []
    
    for i in range(len(order)):
        if time.time() > deadline:
            break
        v = order[i]
        side, x = decode(v)
        
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = (x[c] + y[a] * x[b]) % q
            else:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = (x[c] - x[a] * y[b]) % q
            
            n_val = encode(1 - side, tuple(y))
            
            if n_val not in index:
                index[n_val] = len(order)
                order.append(n_val)
            
            if side == 0:
                edges.append((index[v], index[n_val]))
                
    return len(order), edges
# EVOLVE-BLOCK-END

import time
import math
import random

def construct_lw(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
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

def construct_abs(k, g):
    # Alon-Babai-Suzuki construction on Heisenberg group H_q
    if g % 2 == 0:
        mb = 2 * sum((k-1)**i for i in range(g//2))
    else:
        mb = 1 + k * sum((k-1)**i for i in range((g-1)//2))
        
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0: return False
        return True
        
    q = 2
    while True:
        if is_prime(q):
            if q**3 >= mb:
                break
        q += 1
        
    if k == 4:
        gens = [
            (1, 0, 0),
            (q-1, 0, 0),
            (0, 1, 0),
            (0, q-1, 1)
        ]
    elif k == 6:
        gens = [
            (1, 0, 0),
            (q-1, 0, 0),
            (0, 1, 0),
            (0, q-1, 1),
            (0, 0, 1),
            (0, 0, q-1)
        ]
    else:
        return construct_lw(k, g)

    n = q**3
    edges = []
    def idx(x, y, z):
        return (x * q + y) * q + z
        
    for x in range(q):
        for y in range(q):
            for z in range(q):
                u = idx(x, y, z)
                for gx, gy, gz in gens:
                    vx = (x + gx) % q
                    vy = (y + gy) % q
                    vz = (z + gz + x * gy) % q
                    v = idx(vx, vy, vz)
                    if u < v:
                        edges.append((u, v))
    return n, edges

def construct_cyclic(k, g):
    # Try to find a cyclic k-regular graph of girth g
    # For odd k, use generators +/-g1, +/-g2, g3 (self-inverse)
    # Requires even modulus q for g3 = q/2
    if k % 2 != 0:
        # Estimate q
        if g % 2 == 0:
            mb = 2 * sum((k-1)**i for i in range(g//2))
        else:
            mb = 1 + k * sum((k-1)**i for i in range((g-1)//2))
            
        # Try even q near mb
        start_q = mb if mb % 2 == 0 else mb + 1
        # Search range
        for q in range(start_q, start_q + 500, 2):
            # Generators: 1, q-1, 2, q-2, q/2
            # Check if simple (no loops, no multi-edges)
            # 1 != 0, q-1 != 0, 2 != 0 (q>2), q-2 != 0, q/2 != 0
            # Check no collisions:
            # 1 == q-1 -> q=2 (no)
            # 1 == 2 -> q=1 (no)
            # 1 == q/2 -> q=2 (no)
            # 2 == q/2 -> q=4 (no)
            # So distinct for q > 4.
            # Check girth
            # BFS from 0
            # If shortest cycle < g, fail
            # Optimized: just check if any path of length < g returns to 0
            # This is expensive.
            # Heuristic: assume valid for now if q is large enough?
            # No, must be valid.
            pass
    return None

def construct(k, g):
    if k % 2 == 0:
        return construct_abs(k, g)
    else:
        # Try cyclic if possible, else LW
        # For now, just LW
        return construct_lw(k, g)

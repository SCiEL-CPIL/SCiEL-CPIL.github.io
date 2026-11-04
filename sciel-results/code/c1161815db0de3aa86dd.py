import time
import math

def construct_lw(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned. Valid, far from optimal.
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
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
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
    # Degree k must be even.
    # k=4: generators A, A^-1, B, B^-1
    # k=6: generators A, A^-1, B, B^-1, C, C^-1
    # Find q such that q^3 >= Moore Bound
    
    # Moore bound for k-regular graph of girth g
    if g % 2 == 0:
        # Bipartite lower bound
        # n >= 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
        mb = 2 * sum((k-1)**i for i in range(g//2))
    else:
        # Non-bipartite lower bound
        # n >= 1 + k * sum_{i=0}^{(g-1)/2 - 1} (k-1)^i
        mb = 1 + k * sum((k-1)**i for i in range((g-1)//2))
        
    # Find smallest prime power q >= 2 such that q^3 >= mb
    # For simplicity, we iterate primes
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
        
    # Construct H_q
    # Elements: (x, y, z) in GF(q)^3
    # Multiplication: (x, y, z) * (x', y', z') = (x+x', y+y', z+z' + x*y')
    # Identity: (0, 0, 0)
    
    # Generators
    if k == 4:
        gens = [
            (1, 0, 0),      # A
            (q-1, 0, 0),    # A^-1
            (0, 1, 0),      # B
            (0, q-1, 1)     # B^-1 = (0, -1, 1)
        ]
    elif k == 6:
        gens = [
            (1, 0, 0),      # A
            (q-1, 0, 0),    # A^-1
            (0, 1, 0),      # B
            (0, q-1, 1),    # B^-1
            (0, 0, 1),      # C
            (0, 0, q-1)     # C^-1
        ]
    else:
        # Fallback to LW for other even k
        return construct_lw(k, g)

    n = q**3
    edges = []
    
    # Map (x,y,z) to index
    def idx(x, y, z):
        return (x * q + y) * q + z
        
    # Generate edges
    for x in range(q):
        for y in range(q):
            for z in range(q):
                u = idx(x, y, z)
                for gx, gy, gz in gens:
                    # v = u * g
                    vx = (x + gx) % q
                    vy = (y + gy) % q
                    vz = (z + gz + x * gy) % q
                    v = idx(vx, vy, vz)
                    if u < v:
                        edges.append((u, v))
                        
    return n, edges

def construct(k, g):
    if k % 2 == 0:
        return construct_abs(k, g)
    else:
        return construct_lw(k, g)

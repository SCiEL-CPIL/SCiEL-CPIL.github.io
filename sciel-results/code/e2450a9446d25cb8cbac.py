import math
import time

def construct(k, g):
    # Moore bound for k=3, even g
    r = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(r))
    
    if moore_bound % 2 != 0:
        moore_bound += 1
    
    start_time = time.time()
    deadline = 115  # seconds
    
    best_n = None
    best_a = None
    
    # Try small circulant graphs first
    max_n_circulant = 5000
    
    for n in range(moore_bound, max_n_circulant + 1, 2):
        if time.time() - start_time > deadline:
            break
            
        # Try all valid 'a'
        for a in range(1, n // 2):
            # S = {a, n-a, n/2}
            # Check connectivity: gcd(n, a, n/2) == 1
            if math.gcd(n, a, n // 2) != 1:
                continue
                
            if _circulant_girth_check(n, a, g):
                best_n = n
                best_a = a
                break
        
        if best_n is not None:
            break
    
    if best_n is not None:
        a = best_a
        edges = []
        s1 = a
        s2 = best_n - a
        s3 = best_n // 2
        for u in range(best_n):
            for s in (s1, s2, s3):
                v = (u + s) % best_n
                if u < v:
                    edges.append((u, v))
        return best_n, edges
    
    # If no small circulant found, return LUW fallback
    return _luw_fallback(k, g)

def _circulant_girth_check(n, a, g):
    """
    Check if circulant graph C(n; {a, n-a, n/2}) has girth >= g.
    Uses bitset operations for speed.
    """
    s1 = a
    s2 = n - a
    s3 = n // 2
    
    # R_d = set of positions reachable in exactly d steps
    # Use a bitset where bit i is set if position i is in R_d
    # Start: R_0 = {0}
    R_prev = 1  # bit 0 set
    
    for d in range(1, g):
        # R_curr = R_prev shifted by s1, s2, s3, then OR'd together
        # Bit i in R_curr is set if (i - s1) % n, (i - s2) % n, or (i - s3) % n is in R_prev
        # So R_curr = rotate(R_prev, s1) | rotate(R_prev, s2) | rotate(R_prev, s3)
        
        def rotate(bitset, shift):
            # Rotate bitset right by shift positions (mod n)
            # Bit i in result is bit (i - shift) % n in original
            # Equivalent to: ((bitset << shift) | (bitset >> (n - shift))) & mask
            mask = (1 << n) - 1
            return ((bitset << shift) | (bitset >> (n - shift))) & mask
        
        R_curr = (rotate(R_prev, s1) | rotate(R_prev, s2) | rotate(R_prev, s3))
        
        # Check if 0 is in R_curr
        if R_curr & 1:
            return False
        
        R_prev = R_curr
    
    return True

def _luw_fallback(k, g):
    # The original LUW construction
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

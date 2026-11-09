import time
import random
import sys
import math

def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible.

    Strategy:
    1. For k=3, try to find a Cayley graph on Z_p with high girth.
    2. Fallback to LUW construction if Cayley search fails or times out.
    """
    deadline = time.time() + 115

    if k != 3:
        d = max(3, (g + 1) // 2)
        if 2 * d - 1 < g:
            d += 1
        n, edges = _luw_construct(k, d)
        return n, edges

    # k == 3
    # Try Cayley graph search
    best_n, best_edges = _search_cayley_3_regular(g, deadline)
    
    if best_n > 0:
        return best_n, best_edges

    # Fallback to LUW
    d = (g + 1) // 2
    if 2 * d - 1 < g:
        d += 1
    n_base, edges_base = _luw_construct(3, d)
    return n_base, edges_base

def _search_cayley_3_regular(g, deadline):
    """Search for a 3-regular Cayley graph on Z_p with girth >= g."""
    # Moore bound
    if g % 2 == 0:
        gprime = g // 2
        moore = 2 * sum((3 - 1) ** i for i in range(gprime))
    else:
        gprime = (g - 1) // 2
        moore = 1 + 3 * sum((3 - 1) ** i for i in range(gprime))
    
    if moore % 2 != 0:
        moore += 1
        
    # Start from next prime >= moore
    p = _next_prime(moore)
    
    # Limit search to p < 20000 to avoid timeout, LUW is ~40000
    max_p = 20000
    
    # Precompute (A, B) pairs with hexagonal length < g
    # Length L(A, B) = min |x|+|y|+|z| s.t. x+z=A, y+z=B
    ab_pairs = []
    max_coord = g
    for A in range(-max_coord, max_coord + 1):
        for B in range(-max_coord, max_coord + 1):
            if A == 0 and B == 0:
                continue
            min_len = float('inf')
            for z in range(-max_coord, max_coord + 1):
                x = A - z
                y = B - z
                if abs(x) + abs(y) + abs(z) < min_len:
                    min_len = abs(x) + abs(y) + abs(z)
            if min_len < g:
                ab_pairs.append((A, B))
                
    if not ab_pairs:
        return 0, []

    while p <= max_p:
        if time.time() > deadline:
            break
            
        if _is_prime(p):
            # Search for a, b
            # a in 1..p/3, b in a+1..p/2
            found = False
            step_a = 1
            step_b = 1
            
            for a in range(1, p // 3 + 1, step_a):
                if time.time() > deadline:
                    return 0, []
                for b in range(a + 1, p // 2 + 1, step_b):
                    # Check distinctness conditions
                    if (2 * a + b) % p == 0 or (a + 2 * b) % p == 0:
                        continue
                        
                    # Check girth
                    valid_girth = True
                    for A, B in ab_pairs:
                        if (A * a + B * b) % p == 0:
                            valid_girth = False
                            break
                            
                    if valid_girth:
                        # Found valid graph
                        edges = []
                        for u in range(p):
                            v1 = (u + a) % p
                            if u < v1:
                                edges.append((u, v1))
                            v2 = (u + b) % p
                            if u < v2:
                                edges.append((u, v2))
                            v3 = (u + a + b) % p
                            if u < v3:
                                edges.append((u, v3))
                        return p, edges
                        
        p = _next_prime(p + 1)
        
    return 0, []

def _luw_construct(q, d):
    """Lazebnik-Ustimenko-Woldar construction. Returns (n, edges)."""
    add = (lambda x, y: (x + y) % q)
    sub = (lambda x, y: (x - y) % q)
    mul = (lambda x, y: (x * y) % q)

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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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

def _next_prime(n):
    if n <= 2:
        return 2
    if n % 2 == 0:
        n += 1
    while not _is_prime(n):
        n += 2
    return n

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

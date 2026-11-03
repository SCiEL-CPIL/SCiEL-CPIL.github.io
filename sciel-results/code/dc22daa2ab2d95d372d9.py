import math
import itertools
import numpy as np

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
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

    if k != 4 or g != 10:
        return luw_construct(k, g)

    # Target N=320 (known optimum)
    # Try specific candidates for C_320(a,b)
    # Known good pairs for bipartite circulants often involve small numbers
    candidates = [(1, 3), (1, 5), (1, 7), (1, 9), (1, 11), (1, 13), (1, 15), (1, 17), (1, 19), (1, 21), 
                  (1, 23), (1, 25), (1, 27), (1, 29), (1, 31), (1, 33), (1, 35), (1, 37), (1, 39), (1, 41),
                  (2, 3), (2, 5), (2, 7), (2, 9), (2, 11), (2, 13), (2, 15), (2, 17), (2, 19), (2, 21),
                  (3, 5), (3, 7), (3, 9), (3, 11), (3, 13), (3, 15), (3, 17), (3, 19), (3, 21), (3, 23)]
    
    def check_girth_fast(N, a, b):
        # BFS from each vertex to find shortest cycle
        # Since it's 4-regular, we can use a simple BFS
        # But for N=320, checking 320 BFSs is 320 * 320 * 4 ~ 400k ops. Fast.
        # However, we only need to check if girth < 10.
        # We can stop BFS at depth 5.
        
        adj = [[] for _ in range(N)]
        for i in range(N):
            adj[i].append((i+a)%N)
            adj[i].append((i-a)%N)
            adj[i].append((i+b)%N)
            adj[i].append((i-b)%N)
            
        # Check for cycles of length 4, 6, 8
        # Using the combinatorial method is faster for fixed N, a, b
        # 4^8 = 65536.
        # 320 candidates * 65536 = 20M ops. Might be too slow in Python.
        
        # Let's use the combinatorial check but optimized
        gens = [a, -a, b, -b]
        inv_idx = {0: 1, 1: 0, 2: 3, 3: 2}
        
        # Length 4
        for i1 in range(4):
            for i2 in range(4):
                if i2 == inv_idx[i1]: continue
                for i3 in range(4):
                    if i3 == inv_idx[i2]: continue
                    for i4 in range(4):
                        if i4 == inv_idx[i3]: continue
                        if (gens[i1] + gens[i2] + gens[i3] + gens[i4]) % N == 0:
                            return False
                            
        # Length 6
        for i1 in range(4):
            for i2 in range(4):
                if i2 == inv_idx[i1]: continue
                for i3 in range(4):
                    if i3 == inv_idx[i2]: continue
                    for i4 in range(4):
                        if i4 == inv_idx[i3]: continue
                        for i5 in range(4):
                            if i5 == inv_idx[i4]: continue
                            for i6 in range(4):
                                if i6 == inv_idx[i5]: continue
                                if (gens[i1] + gens[i2] + gens[i3] + gens[i4] + gens[i5] + gens[i6]) % N == 0:
                                    return False
                                    
        # Length 8
        for i1 in range(4):
            for i2 in range(4):
                if i2 == inv_idx[i1]: continue
                for i3 in range(4):
                    if i3 == inv_idx[i2]: continue
                    for i4 in range(4):
                        if i4 == inv_idx[i3]: continue
                        s4 = gens[i1] + gens[i2] + gens[i3] + gens[i4]
                        for i5 in range(4):
                            if i5 == inv_idx[i4]: continue
                            s5 = s4 + gens[i5]
                            for i6 in range(4):
                                if i6 == inv_idx[i5]: continue
                                s6 = s5 + gens[i6]
                                for i7 in range(4):
                                    if i7 == inv_idx[i6]: continue
                                    s7 = s6 + gens[i7]
                                    for i8 in range(4):
                                        if i8 == inv_idx[i7]: continue
                                        if (s7 + gens[i8]) % N == 0:
                                            return False
        return True

    N_target = 320
    for a, b in candidates:
        if a == b or (a+b) % N_target == 0 or (a-b) % N_target == 0:
            continue
        if math.gcd(math.gcd(a, b), N_target) != 1:
            continue
        if check_girth_fast(N_target, a, b):
            edges = []
            for i in range(N_target):
                edges.append((i, (i + a) % N_target))
                edges.append((i, (i + b) % N_target))
            return N_target, edges
            
    return luw_construct(k, g)

import math
import itertools

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

    # Try to find a circulant graph C_N(a, b) with girth >= 10
    # Search N from lower bound 243 to 500
    # Generators a, b
    
    def check_girth(N, a, b):
        # Generators: a, -a, b, -b
        # Check for non-backtracking cycles of length 4, 6, 8
        gens = [a, -a, b, -b]
        # Map generator to its inverse index for backtracking check
        # 0: a, 1: -a, 2: b, 3: -b
        # Inverse of 0 is 1, 1 is 0, 2 is 3, 3 is 2
        inv_idx = {0: 1, 1: 0, 2: 3, 3: 2}
        
        # Length 4
        # w1, w2, w3, w4
        # w2 != -w1, w3 != -w2, w4 != -w3
        # Sum == 0
        # We can iterate over indices
        for i1 in range(4):
            for i2 in range(4):
                if i2 == inv_idx[i1]: continue
                for i3 in range(4):
                    if i3 == inv_idx[i2]: continue
                    for i4 in range(4):
                        if i4 == inv_idx[i3]: continue
                        s = gens[i1] + gens[i2] + gens[i3] + gens[i4]
                        if s % N == 0:
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
                                s = gens[i1] + gens[i2] + gens[i3] + gens[i4] + gens[i5] + gens[i6]
                                if s % N == 0:
                                    return False
                                    
        # Length 8
        # This is 4^8 / 4 (approx) = 65536 * 4 = 260k iterations.
        # Might be slow in Python if done for many N, a, b.
        # Optimization: Precompute sums of length 4?
        # Or just run it. 260k is fast.
        for i1 in range(4):
            for i2 in range(4):
                if i2 == inv_idx[i1]: continue
                for i3 in range(4):
                    if i3 == inv_idx[i2]: continue
                    for i4 in range(4):
                        if i4 == inv_idx[i3]: continue
                        s4 = gens[i1] + gens[i2] + gens[i3] + gens[i4]
                        # Check if s4 can be extended to 0 in 4 more steps
                        # We need w5, w6, w7, w8 such that s4 + sum(w5..w8) = 0
                        # And non-backtracking constraints
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
                                        s8 = s7 + gens[i8]
                                        if s8 % N == 0:
                                            return False
        return True

    # Search for circulant graph C_N(a, b) with girth >= 10
    # We only need to check for cycles of length 4, 6, 8.
    # If none exist, girth is at least 10.
    
    def has_short_cycle(N, a, b):
        gens = [a, -a, b, -b]
        
        # Length 4
        for i1 in range(4):
            for i2 in range(4):
                if i2 == (i1 ^ 1): continue
                s12 = gens[i1] + gens[i2]
                for i3 in range(4):
                    if i3 == (i2 ^ 1): continue
                    s123 = s12 + gens[i3]
                    for i4 in range(4):
                        if i4 == (i3 ^ 1): continue
                        if (s123 + gens[i4]) % N == 0:
                            return True
        
        # Length 6
        for i1 in range(4):
            for i2 in range(4):
                if i2 == (i1 ^ 1): continue
                s12 = gens[i1] + gens[i2]
                for i3 in range(4):
                    if i3 == (i2 ^ 1): continue
                    s123 = s12 + gens[i3]
                    for i4 in range(4):
                        if i4 == (i3 ^ 1): continue
                        s1234 = s123 + gens[i4]
                        for i5 in range(4):
                            if i5 == (i4 ^ 1): continue
                            s12345 = s1234 + gens[i5]
                            for i6 in range(4):
                                if i6 == (i5 ^ 1): continue
                                if (s12345 + gens[i6]) % N == 0:
                                    return True
                                    
        # Length 8
        for i1 in range(4):
            for i2 in range(4):
                if i2 == (i1 ^ 1): continue
                s12 = gens[i1] + gens[i2]
                for i3 in range(4):
                    if i3 == (i2 ^ 1): continue
                    s123 = s12 + gens[i3]
                    for i4 in range(4):
                        if i4 == (i3 ^ 1): continue
                        s1234 = s123 + gens[i4]
                        for i5 in range(4):
                            if i5 == (i4 ^ 1): continue
                            s12345 = s1234 + gens[i5]
                            for i6 in range(4):
                                if i6 == (i5 ^ 1): continue
                                s123456 = s12345 + gens[i6]
                                for i7 in range(4):
                                    if i7 == (i6 ^ 1): continue
                                    s1234567 = s123456 + gens[i7]
                                    for i8 in range(4):
                                        if i8 == (i7 ^ 1): continue
                                        if (s1234567 + gens[i8]) % N == 0:
                                            return True
        return False

    import time
    start_time = time.time()
    
    # Search N from Moore bound 161 to 500
    for N in range(161, 501):
        if time.time() - start_time > 100:
            break
        for a in range(1, N // 2 + 1):
            if 2 * a % N == 0: continue
            for b in range(a + 1, N // 2 + 1):
                if 2 * b % N == 0: continue
                if (a + b) % N == 0: continue
                if (a - b) % N == 0: continue
                if math.gcd(math.gcd(a, b), N) != 1: continue
                
                if not has_short_cycle(N, a, b):
                    edges = []
                    for i in range(N):
                        edges.append((i, (i + a) % N))
                        edges.append((i, (i + b) % N))
                    return N, edges
                    
    return luw_construct(k, g)

import time
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140

    # If the task is for 4-regular graph of girth 10
    if k == 4 and g == 10:
        # Try to find a circulant graph C_n(a, b) with girth >= 10
        # n from 242 (Moore bound) to 320 (best known)
        
        def has_short_cycle(n, a, b):
            # Check for cycles of length 3 and 4 using linear combinations
            # Steps are +-a, +-b
            # L=3: 3a, 3b, 2a-b, 2b-a, a+b, a-b
            for s in [3*a, 3*b, 2*a-b, 2*b-a, a+b, a-b]:
                if s % n == 0:
                    return True
            
            # L=4: 4a, 4b, 3a-b, 3b-a, 2a+b, 2b+a, 2a-b (done), a+b (done)
            # Also a-a-b-b? No, backtracking.
            # Non-backtracking walks of length 4.
            # We'll rely on matrix for L>=5, but L=4 is cheap to check manually?
            # Actually, matrix check for L=4 is also fast.
            
            # Build adjacency matrix
            A = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                A[i, (i+a)%n] = 1
                A[i, (i-a)%n] = 1
                A[i, (i+b)%n] = 1
                A[i, (i-b)%n] = 1
            
            # Check L=4 to 9
            A_k = A.copy()
            for L in range(4, 10):
                A_k = A_k @ A
                if np.any(np.diag(A_k) > 0):
                    return True
            return False

        # Search
        # We can randomize the order of n to find a solution faster if one exists in the middle
        # But sequential is safer.
        # Let's try n=320 down to 242.
        
        for n in range(320, 241, -1):
            # Prune a, b
            # a, b must be coprime to n? No, just distinct.
            # a != b, a != n-b, etc.
            for a in range(1, n//2 + 1):
                for b in range(a+1, n//2 + 1):
                    # Skip if a+b == n or a-b == 0 etc.
                    if (a + b) % n == 0 or (a - b) % n == 0:
                        continue
                    if has_short_cycle(n, a, b):
                        continue
                    
                    # Found a valid graph
                    edges = []
                    seen = set()
                    for i in range(n):
                        for j in [(i+a)%n, (i-b)%n]:
                            u, v = min(i, j), max(i, j)
                            if (u, v) not in seen:
                                seen.add((u, v))
                                edges.append((u, v))
                    
                    return n, edges
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
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
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

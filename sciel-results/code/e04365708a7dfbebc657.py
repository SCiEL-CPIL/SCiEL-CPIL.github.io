import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100
    
    if k != 3 or g != 16:
        # Fallback to seed
        pass # Implement seed fallback below

    def check_circulant(n, a):
        S = [a, (-a) % n, n // 2]
        # Unique S
        S = list(set(S))
        if len(S) < 3:
            return False
        if 0 in S:
            return False
            
        # reach[l] is set of sums of l steps
        reach = [set() for _ in range(g)]
        reach[0] = {0}
        
        for l in range(1, g):
            curr = set()
            for s in S:
                for prev in reach[l-1]:
                    curr.add((prev + s) % n)
            if 0 in curr:
                # Check if it's a valid cycle (length >= 3)
                # If l >= 3, it's a cycle.
                # If l < 3, it's a walk, not a simple cycle (in simple graph).
                # But if l=2, 0->s1->s2->0. s1+s2=0. s2=-s1.
                # This is just backtracking. Not a cycle.
                # So we only care if l >= 3.
                if l >= 3:
                    return False
            reach[l] = curr
            
        return True

    # Search
    for n in range(512, 1200, 2):
        if time.time() > deadline:
            break
        # Sample some a's
        # a must be coprime to n? No.
        # But a should not be 0 or n/2.
        candidates = list(range(1, n//2))
        random.shuffle(candidates)
        # Check first 50
        for a in candidates[:50]:
            if check_circulant(n, a):
                edges = []
                S = [a, (-a) % n, n // 2]
                for i in range(n):
                    for s in S:
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges

    # Fallback: Seed logic
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

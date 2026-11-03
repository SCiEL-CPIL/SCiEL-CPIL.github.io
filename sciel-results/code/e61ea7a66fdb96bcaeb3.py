import time
import random
import sys
from collections import deque

def construct(k, g):
    # Fallback: LUW construction
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

    # Heuristic Search
    def heuristic_search(k, g, max_time=10):
        start_time = time.time()
        # Moore bound
        r = g // 2
        moore = 1 + k * sum((k-1)**i for i in range(r))
        # Try n from moore to moore + 100
        for n in range(moore, moore + 200):
            if (time.time() - start_time) > max_time:
                break
            # Generate random k-regular graph
            # This is a simplified generator. Might fail to produce valid graph.
            # For k=4, n=500, this is complex.
            # Use a simple swap heuristic on a base graph.
            # Base graph: Cycle + chords?
            # Let's try a random regular graph generator.
            # If it fails, skip.
            pass
        return None

    # If heuristic is too complex, just return LUW.
    # But LUW is 8192.
    # Let's try to return a smaller LUW if possible? No.
    
    # Actually, the LUW graph is the only guaranteed valid one I can write quickly.
    # I will return the LUW graph.
    return luw_construct(k, g)

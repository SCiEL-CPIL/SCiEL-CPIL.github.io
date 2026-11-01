import numpy as np
import time
import random

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Fallback: LUW Construction from seed
    def luw_construct():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
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

    # Target: Find a small graph.
    # For k=4, g=10, the cage is 102 vertices (Biggs-Smith).
    # We try to find it via local search.
    if k == 4 and g == 10:
        n_target = 102
        # Initialize random 4-regular graph
        # Use a simple method: random permutation of neighbors
        def random_4_reg(n):
            # Create a multigraph first, then remove duplicates if possible
            # Simple approach: random pairing of half-edges
            half_edges = []
            for i in range(n):
                for _ in range(k):
                    half_edges.append(i)
            random.shuffle(half_edges)
            adj = {i: [] for i in range(n)}
            for i in range(0, n * k, 2):
                u, v = half_edges[i], half_edges[i+1]
                if u != v and v not in adj[u]:
                    adj[u].append(v)
                    adj[v].append(u)
                else:
                    # Retry or fix later
                    pass
            # Ensure degree 4
            # This simple method often fails. Use a better one.
            # Use networkx if available? No, std lib only.
            # Use a known 4-regular graph construction:
            # Cycle C_n + chords?
            # For n=102, we can use a circulant graph.
            # C_102(1, 2, 3, 4)? No, that's 8-regular.
            # C_102(1, 51)? No.
            # Let's just use the LUW fallback if we can't easily build a good candidate.
            return None

        # Given the complexity of finding the exact 102-vertex graph via heuristic in 150s,
        # and the risk of returning an invalid graph, we stick to the LUW construction
        # but verify if we can improve the seed's performance.
        # The seed's LUW construction is deterministic.
        # We return the LUW result.
        return luw_construct()
    
    # For other parameters, use LUW
    return luw_construct()

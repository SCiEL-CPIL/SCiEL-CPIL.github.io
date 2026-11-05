# EVOLVE-BLOCK-START
import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    # --- LUW Construction ---
    def luw_construct(k, g):
        q = k
        if g <= 5:
            d = 3
        else:
            d = 3
            while True:
                girth_est = d + 5 if d % 2 == 1 else d + 4
                if girth_est >= g:
                    break
                d += 1

        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        rule_a = [0] * (d + 1)
        rule_b = [0] * (d + 1)
        for c in range(2, d + 1):
            rule_a[c], rule_b[c] = rule[c]

        mul_table = [[mul(i, j) for j in range(q)] for i in range(q)]
        add_table = [[add(i, j) for j in range(q)] for i in range(q)]
        sub_table = [[sub(i, j) for j in range(q)] for i in range(q)]

        def compute_neighbors_point(x):
            result = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = add_table[x[c]][mul_table[y[a]][x[b]]]
                result.append((1, tuple(y)))
            return result

        def compute_neighbors_line(x):
            result = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = sub_table[x[c]][mul_table[x[a]][y[b]]]
                result.append((0, tuple(y)))
            return result

        start = (0, (0,) * (d + 1))
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            if time.time() > deadline - 10:
                break
            side, x = vertex
            if side == 0:
                neighbors = compute_neighbors_point(x)
            else:
                neighbors = compute_neighbors_line(x)
            for other in neighbors:
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if side == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    n_luw, edges_luw = luw_construct(k, g)

    # --- Try to verify and potentially use a smaller known construction ---
    # For k=4, g=10, the known best is 320. The LUW construction with d=5, q=4 gives:
    # d=5 (odd), girth_est = 5+5 = 10 >= 10, so d=5.
    # The LUW graph CD(5,4) has 2 * 4^5 = 2048 vertices? No, it's a bipartite graph.
    # Actually, the LUW graph CD(d,q) has 2*q^d vertices. For d=5, q=4, that's 2*4^5 = 2048.
    # But we only take the connected component of the zero vertex.
    # For q=4, d=5, the connected component might be much smaller.
    # Let's just use the LUW result for now.

    return n_luw, edges_luw
# EVOLVE-BLOCK-END

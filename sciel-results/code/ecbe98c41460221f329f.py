def construct(k, g):
    import time
    from collections import deque

    start_time = time.time()
    deadline = start_time + 115

    q = k

    def build_luw(d, q):
        if q == 4:
            mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            def field_add(a, b):
                return a ^ b
            def field_sub(a, b):
                return a ^ b
            def field_mul(a, b):
                return mul_table[a][b]
        else:
            def field_add(a, b):
                return (a + b) % q
            def field_sub(a, b):
                return (a - b) % q
            def field_mul(a, b):
                return (a * b) % q

        rules = [None] * (d + 1)
        rules[2] = (1, 1)
        rules[3] = (2, 1)
        rules[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            if r == 0:
                rules[c] = (1, c0 - 2)
            elif r == 1:
                rules[c] = (c0 - 1, 1)
            elif r == 2:
                rules[c] = (c0, 1)
            else:
                rules[c] = (1, c0 + 1)

        start = (0,) + (0,) * d
        index = {start: 0}
        order = [start]
        edges = []

        for vi, vertex in enumerate(order):
            side = vertex[0]
            x = vertex[1:]

            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rules[c]
                    if side == 0:
                        y[c] = field_add(x[c-1], field_mul(y[a-1], x[b-1]))
                    else:
                        y[c] = field_sub(x[c-1], field_mul(x[a-1], y[b-1]))

                other = (1 - side,) + tuple(y[1:])

                if other not in index:
                    index[other] = len(order)
                    order.append(other)

                if side == 0:
                    edges.append((vi, index[other]))

        return len(order), edges

    d_min = max(3, (g - 1) // 2)

    best_n = None
    best_edges = None

    for d in range(d_min, d_min + 5):
        if time.time() > deadline - 10:
            break
        try:
            n, edges = build_luw(d, q)
            if n > 0 and n < 1000000:
                deg_counts = [0] * n
                for u, v in edges:
                    deg_counts[u] += 1
                    deg_counts[v] += 1
                if all(c == k for c in deg_counts):
                    if best_n is None or n < best_n:
                        best_n, best_edges = n, edges
                    if n <= 1300:
                        break
        except Exception:
            continue

    if best_n is None:
        d = d_min
        n, edges = build_luw(d, q)
        best_n, best_edges = n, edges

    return best_n, best_edges

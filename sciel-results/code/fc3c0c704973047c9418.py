"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct(k, g):
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    # Girth guarantee: d+5 if d odd, d+4 if d even.
    # Choose minimal d such that guaranteed girth >= g.
    q = k

    # Find minimal d >= 3 with girth guarantee >= g
    d = 3
    while True:
        girth_est = d + 5 if d % 2 == 1 else d + 4
        if girth_est >= g:
            break
        d += 1

    # GF(q) operations
    if q == 4:
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return mul_table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # Precompute rules for the LUW construction
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    rule_a[2], rule_b[2] = 1, 1
    rule_a[3], rule_b[3] = 2, 1
    rule_a[4], rule_b[4] = 1, 2
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            a, b = 1, c0 - 2
        elif r == 1:
            a, b = c0 - 1, 1
        elif r == 2:
            a, b = c0, 1
        else:
            a, b = 1, c0 + 1
        rule_a[c] = a
        rule_b[c] = b

    # Precompute multiplication tables for speed
    # Precompute all mul(y[a], x[b]) combinations is too much; instead,
    # we'll just use the mul function which is fast enough.

    def compute_neighbours(vertex):
        side, x = vertex
        result = []
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            result.append((1 - side, tuple(y)))
        return result

    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []

    # BFS to build the connected component
    from collections import deque
    queue = deque([start])
    head = 0

    while head < len(order):
        if time.time() > deadline:
            break
        vertex = order[head]
        head += 1

        for other in compute_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge if from Point side (0) to avoid duplicates
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END

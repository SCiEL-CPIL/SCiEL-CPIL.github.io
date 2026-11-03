import time
import collections

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _luw_construct(k, g):
    """
    LUW construction for k-regular graphs.
    Girth is 2d + 2.
    We need 2d + 2 >= g => d >= (g - 2) / 2.
    So d = ceil((g - 2) / 2).
    For integer g, d = (g - 2 + 1) // 2 = (g - 1) // 2.
    """
    # Calculate minimal d
    d = (g - 1) // 2
    if d < 1:
        d = 1
        
    # LUW parameters
    q = k
    # Note: Standard LUW uses field of size q.
    # For k=3, q=3.
    
    # Multiplication table for GF(q) if needed, but for prime q, simple mod works.
    # For k=3, q=3 is prime.
    # For k=4, q=4 is not prime, need GF(4).
    # The original code handled k=4 with XOR.
    # We will keep the general structure.

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Rule for LUW
    # The rule defines the recurrence.
    # For girth 2d+2, the standard rule is used.
    rule = {}
    # Base cases
    if d >= 2:
        rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
        
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        # The pattern in the original code:
        # [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
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
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        if _time_left() < 5.0:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0

    n_luw, edges_luw = _luw_construct(k, g)

    return n_luw, edges_luw

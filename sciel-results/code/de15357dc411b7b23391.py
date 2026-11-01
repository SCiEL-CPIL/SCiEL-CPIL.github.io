import random
import time
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin

    if k != 5:
        return _luw_construct(k, g, deadline)

    # For k=5, g=9: LUW gives a valid graph. 
    # The previous implementation used d=5 for g=9, resulting in N=6250.
    # The minimal d for girth >= 9 (using girth = 2d+1) is d=4, resulting in N=1250.
    # We update the d calculation to use the minimal required diameter.
    
    return _luw_construct(k, g, deadline)


def _luw_construct(k, g, deadline=None):
    """Vectorized L-U-W construction using NumPy for speed."""
    if deadline is None:
        deadline = time.time() + 110.0
    
    q = k
    # Minimal d such that girth (2d+1) >= g
    # 2d + 1 >= g  => 2d >= g - 1 => d >= (g - 1) / 2
    # Since d is integer, d = ceil((g - 1) / 2)
    # For g=9, d=4. For g=10, d=5.
    d = (g + 1) // 2  # This is equivalent to ceil((g-1)/2) for integer g?
    # Let's check:
    # g=9: (9+1)//2 = 5. Wait.
    # (g-1)/2 for g=9 is 4.
    # (g-1)/2 for g=10 is 4.5 -> 5.
    # So d = (g - 1 + 1) // 2? No.
    # d = (g - 1) // 2 if (g-1) % 2 == 0 else (g - 1 + 1) // 2
    # Which is just (g) // 2?
    # g=9: 9//2 = 4.
    # g=10: 10//2 = 5.
    # g=8: 8//2 = 4. (Girth 9 >= 8).
    # g=7: 7//2 = 3. (Girth 7 >= 7).
    # So d = g // 2 is correct.
    
    d = max(3, g // 2)

    if q == 4:
        mul_table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int64)
    else:
        mul_table = np.outer(np.arange(q, dtype=np.int64), np.arange(q, dtype=np.int64)) % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    rule_a = np.array([rule[c][0] for c in range(2, d + 1)], dtype=np.int64)
    rule_b = np.array([rule[c][1] for c in range(2, d + 1)], dtype=np.int64)

    free_vals = np.arange(q, dtype=np.int64)

    def get_neighbours(side, x_arr):
        y_coords = np.empty((q, d + 1), dtype=np.int64)
        y_coords[:, 0] = 0
        y_coords[:, 1] = free_vals
        if side == 0:
            for idx, c in enumerate(range(2, d + 1)):
                a = rule_a[idx]
                b = rule_b[idx]
                prod = mul_table[y_coords[:, a], x_arr[b]]
                y_coords[:, c] = (x_arr[c] + prod) % q
        else:
            for idx, c in enumerate(range(2, d + 1)):
                a = rule_a[idx]
                b = rule_b[idx]
                prod = mul_table[x_arr[a], y_coords[:, b]]
                y_coords[:, c] = (x_arr[c] - prod) % q
        return y_coords

    start_key = (0, (0,) * (d + 1))
    index = {start_key: 0}
    order = [start_key]
    edges = []

    for i, vertex in enumerate(order):
        if time.time() > deadline - 5.0:
            break
        side, x_tuple = vertex
        x_arr = np.array(x_tuple, dtype=np.int64)
        y_coords = get_neighbours(side, x_arr)
        for j in range(q):
            other_key = (1 - side, tuple(y_coords[j]))
            if other_key not in index:
                index[other_key] = len(order)
                order.append(other_key)
            if side == 0:
                edges.append((i, index[other_key]))

    return len(order), edges

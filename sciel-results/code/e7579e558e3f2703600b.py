import time
import numpy as np
from math import gcd

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    if k == 4 and g == 10:
        # Moore bound for (4,10): 216
        # Best known: 320
        
        # 1. Try to find a circulant graph C(n, {1, s})
        # This is fast and might find a solution if one exists (though (4,10) cage is not circulant)
        best_n = None
        best_edges = None
        
        for n in range(216, 325):
            if time.time() > deadline - 20:
                break
            for s in range(2, n // 2 + 1):
                if time.time() > deadline - 20:
                    break
                if gcd(s, n) != 1:
                    continue
                if _check_girth_circulant_fast(n, 1, s, g):
                    best_n = n
                    best_edges = _get_circulant_edges(n, [1, s])
                    break
            if best_n is not None:
                break
        
        if best_n is not None:
            return best_n, best_edges
        
        # 2. Fallback to LUW construction
        # LUW for k=4, g=10 produces n=512
        # This is better than failing, but not optimal.
        # However, we can try to construct the known (4,10)-cage if we had the code.
        # Since we don't, we return LUW.
        
        return _luw_construct(k, g)

    return _luw_construct(k, g)

def _check_girth_circulant_fast(n, s1, s2, g):
    visited = np.full((n, 4), -1, dtype=np.int32)
    queue = []
    
    for d in range(4):
        if d == 0:
            np_ = s1 % n
        elif d == 1:
            np_ = (-s1) % n
        elif d == 2:
            np_ = s2 % n
        else:
            np_ = (-s2) % n
        visited[np_, d] = 1
        queue.append((np_, d, 1))
        
    qi = 0
    max_depth = g - 1
    opp = [1, 0, 3, 2]
    
    while qi < len(queue):
        pos, prev_dir, depth = queue[qi]
        qi += 1
        if depth >= max_depth:
            continue
        for next_dir in range(4):
            if next_dir == opp[prev_dir]:
                continue
            if next_dir == 0:
                np_ = (pos + s1) % n
            elif next_dir == 1:
                np_ = (pos - s1) % n
            elif next_dir == 2:
                np_ = (pos + s2) % n
            else:
                np_ = (pos - s2) % n
            new_depth = depth + 1
            if np_ == 0:
                if new_depth < g:
                    return False
                continue
            if visited[np_, next_dir] != -1:
                if visited[np_, next_dir] <= new_depth:
                    continue
            else:
                visited[np_, next_dir] = new_depth
                queue.append((np_, next_dir, new_depth))
    return True

def _get_circulant_edges(n, steps):
    edges = []
    for i in range(n):
        for s in steps:
            j = (i + s) % n
            if i < j:
                edges.append((i, j))
    return edges

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for k-regular girth-g graphs.
    """
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
            yield (1 - side, tuple(y))
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

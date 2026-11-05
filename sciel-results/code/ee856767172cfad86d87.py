import time
import random
import numpy as np
from math import gcd

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    if k == 4 and g == 10:
        # Search for circulant graphs C_n({1, s})
        # n must be such that a 4-regular graph exists.
        # Start from the known lower bound / optimal size.
        n_start = 320
        max_n = 1000 # Search up to 1000, if not found, use LUW (2048)
        
        for n in range(n_start, max_n + 1):
            if time.time() > deadline:
                break
                
            # Search for s such that C_n({1, s}) has girth >= g
            # s must be coprime to n for the graph to be connected.
            # We can assume s in [2, n//2] due to symmetry.
            for s in range(2, n // 2 + 1):
                if gcd(s, n) != 1:
                    continue
                # Quick checks for small cycles
                # No 4-cycles: 2s != 0, 2s != 1, 2s != n-1
                if 2 * s % n in [0, 1, n - 1]:
                    continue
                # No 6-cycles: 3s != 0, 3s != 1, 3s != n-1, 2s+1 != 0, etc.
                # Let's just run the full girth check, it's fast enough.
                
                if _check_girth_circulant_fast(n, 1, s, g):
                    edges = _get_circulant_edges(n, [1, s])
                    return n, edges
                    
        # Fallback to LUW construction (n=2048)
        return _luw_construct(k, g)

    # General fallback
    return _luw_construct(k, g)

def _check_girth_circulant_fast(n, s1, s2, g):
    """
    Check if C_n({s1, s2}) has girth >= g.
    """
    # Directions: 0:+s1, 1:-s1, 2:+s2, 3:-s2
    # Backtracking pairs: (0,1), (1,0), (2,3), (3,2)
    
    # visited[pos][dir] = min depth
    # We use a list of dicts or a 2D array.
    # Since n is small, a 2D array is fine.
    visited = [[-1 for _ in range(4)] for _ in range(n)]
    
    # Queue: (pos, dir, depth)
    queue = []
    
    # Initial moves from 0
    offsets = [s1, -s1, s2, -s2]
    for d in range(4):
        np_ = (0 + offsets[d]) % n
        visited[np_][d] = 1
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
                
            np_ = (pos + offsets[next_dir]) % n
            new_depth = depth + 1
            
            if np_ == 0:
                if new_depth < g:
                    return False
                else:
                    continue
                    
            if visited[np_][next_dir] != -1:
                if visited[np_][next_dir] <= new_depth:
                    continue
            else:
                visited[np_][next_dir] = new_depth
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

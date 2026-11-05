import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if k == 4 and g == 10:
        n = 320
        # Search over step pairs (s1, s2) with 1 <= s1 < s2 <= n//2
        for s1 in range(1, 20):
            if time.time() > deadline:
                break
            for s2 in range(s1 + 1, n // 2 + 1):
                if time.time() > deadline:
                    break
                if _check_circulant_girth_fast(n, [s1, s2], 4, 10):
                    return n, _get_circulant_edges(n, [s1, s2])
        
        return _luw_construct(k, g)
    
    return _luw_construct(k, g)

def _check_circulant_girth_fast(n, steps, k, g):
    if len(steps) * 2 != k:
        return False
    
    # Quick pre-filters for small cycles
    s0, s1 = steps[0], steps[1]
    
    # Triangle: a+a+b=0, a+b+b=0, a+b+c=0 (but only 2 steps here)
    # 3-cycle: x+y+z=0 mod n with no backtracking
    # Possible: s0+s0+s1=0, s0+s1+s1=0, s0+s0-s1=0, etc.
    # Let's just do the BFS properly
    
    max_len = g - 1
    min_len = {}
    queue = deque()
    
    for e in range(4):
        s_idx = e // 2
        sign = 1 if e % 2 == 0 else -1
        s = steps[s_idx]
        new_v = (0 + sign * s) % n
        state = (new_v, e)
        min_len[state] = 1
        queue.append((new_v, e, 1))
    
    while queue:
        v, last_e, length = queue.popleft()
        
        if length >= max_len:
            continue
        
        for e in range(4):
            # Backtracking check
            if (e // 2) == (last_e // 2) and (e % 2) != (last_e % 2):
                continue
            
            s_idx = e // 2
            sign = 1 if e % 2 == 0 else -1
            s = steps[s_idx]
            new_v = (v + sign * s) % n
            new_length = length + 1
            
            if new_v == 0 and new_length < g:
                return False
            
            state = (new_v, e)
            if state not in min_len or min_len[state] > new_length:
                min_len[state] = new_length
                if new_length < max_len:
                    queue.append((new_v, e, new_length))
    
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
    # Original seed logic
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

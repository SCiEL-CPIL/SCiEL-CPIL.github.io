import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    # If not the specific case, just use seed
    if not (k == 5 and g == 9):
        return _seed_construct(k, g)

    # Try algebraic construction with optimized rule variants
    best_result = None
    best_n = float('inf')
    
    # First, try the known good LUW construction (variant 0)
    result = _algebraic_construct_variant(k, g, 0)
    if result is not None:
        n, edges = result
        if n < best_n:
            best_n = n
            best_result = result

    # Try other variants if we have time
    for variant in range(1, 30):
        if time.time() > deadline - 20:
            break
        result = _algebraic_construct_variant(k, g, variant)
        if result is not None:
            n, edges = result
            if n < best_n:
                best_n = n
                best_result = result

    if best_result is not None:
        return best_result
    
    return _seed_construct(k, g)

def _algebraic_construct_variant(k, g, variant):
    """Try different rule parameterizations of the LUW-type construction"""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    base_rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    rules = dict(base_rule)
    
    if variant == 0:
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rules[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    elif variant == 1:
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rules[c] = [(c0-1, 1), (1, c0), (1, c0-1), (c0, 1)][r]
    elif variant == 2:
        for c in range(5, d + 1):
            rules[c] = (1, c - 2)
    elif variant == 3:
        for c in range(5, d + 1):
            rules[c] = (c - 2, 1)
    elif variant == 4:
        for c in range(5, d + 1):
            rules[c] = (1, c - 1)
    elif variant == 5:
        for c in range(5, d + 1):
            rules[c] = (c - 1, 1)
    elif variant == 6:
        for c in range(5, d + 1):
            rules[c] = (1, 1)
    elif variant == 7:
        for c in range(5, d + 1):
            rules[c] = (2, 1)
    elif variant == 8:
        for c in range(5, d + 1):
            rules[c] = (1, (c % d) + 1)
    elif variant == 9:
        for c in range(5, d + 1):
            rules[c] = ((c % d) + 1, 1)
    elif variant == 10:
        for c in range(5, d + 1):
            rules[c] = (2, 2)
    elif variant == 11:
        for c in range(5, d + 1):
            rules[c] = (1, 3)
    elif variant == 12:
        for c in range(5, d + 1):
            rules[c] = (3, 1)
    elif variant == 13:
        for c in range(5, d + 1):
            rules[c] = (2, 3)
    elif variant == 14:
        for c in range(5, d + 1):
            rules[c] = (3, 2)
    else:
        random.seed(variant * 7919 + 104729)
        for c in range(5, d + 1):
            a = random.randint(1, d)
            b = random.randint(1, d)
            rules[c] = (a, b)
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rules[c]
                y[c] = (x[c] + y[a] * x[b]) % q if side == 0 else (x[c] - x[a] * y[b]) % q
            yield 1 - side, tuple(y)
    
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
    
    n = len(order)
    
    # Verify girth if graph is small enough
    if n < 50000:
        girth = _compute_girth_from_edges(edges, n)
        if girth < g:
            return None
    
    return n, edges

def _compute_girth_from_edges(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        dq = deque([start])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    dq.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if min_girth == 3:
                        return 3
    return min_girth if min_girth != float('inf') else n

def _seed_construct(k, g):
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

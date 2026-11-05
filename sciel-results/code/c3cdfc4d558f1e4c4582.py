import numpy as np
import time
import random
import math
import sys

def construct(k, g):
    if k != 3 or g != 16:
        return _luw_fallback(k, g)
    
    # Petersen Graph Definition
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Map edge to index
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    # Adjacency list with edge indices
    adj = [[] for _ in range(10)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    # Generate all closed non-backtracking walks of length L in [3, g-1]
    max_L = g - 1
    walks = []
    
    for start_v in range(10):
        stack = [(start_v, -1, 0, [])]
        while stack:
            current, prev_edge, length, path = stack.pop()
            
            if length >= 3 and current == start_v:
                walks.append(tuple(path))
                
            if length == max_L:
                continue
                
            for next_v, e_idx in adj[current]:
                if e_idx == prev_edge:
                    continue
                u, v = edges[e_idx]
                if current == u:
                    d = 1
                else:
                    d = -1
                stack.append((next_v, e_idx, length + 1, path + [(e_idx, d)]))
                
    # Convert walks to a constraint matrix
    unique_vectors = set()
    for w in walks:
        vec = [0] * 15
        for e_idx, d in w:
            vec[e_idx] += d
        if any(v != 0 for v in vec):
            unique_vectors.add(tuple(vec))
            
    if not unique_vectors:
        return _luw_fallback(k, g)
        
    C = np.array(list(unique_vectors), dtype=np.int64)
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_m = None
    best_volts = None
    
    min_m = 52
    batch_size = 100000
    
    for m in range(min_m, 2000):
        if time.time() > deadline:
            break
            
        found = False
        max_attempts_per_m = 10 
        for _ in range(max_attempts_per_m):
            if time.time() > deadline:
                break
                
            volts = np.random.randint(1, m, size=(batch_size, 15), dtype=np.int64)
            S = volts @ C.T
            mods = S % m
            valid_mask = np.all(mods != 0, axis=1)
            
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                for idx in valid_indices[:100]:
                    v = volts[idx]
                    g_val = m
                    for val in v:
                        g_val = math.gcd(g_val, int(val))
                        if g_val == 1:
                            break
                    if g_val == 1:
                        found = True
                        best_m = m
                        best_volts = v
                        break
                if found:
                    break
                    
        if found:
            break
            
    if best_m is None:
        return _luw_fallback(k, g)
        
    n = 10 * best_m
    edges_out = []
    
    for e_idx, (u, v) in enumerate(edges):
        x_val = int(best_volts[e_idx])
        for k in range(best_m):
            v1 = u * best_m + k
            v2 = v * best_m + (k + x_val) % best_m
            edges_out.append((v1, v2))
            
    return n, edges_out

def _luw_fallback(k, g):
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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    return len(order), edges_luw

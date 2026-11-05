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
    # g=16, so L in [3, 15]
    max_L = g - 1
    walks = []
    
    # DFS to find closed walks
    # We generate from all vertices to ensure we capture all cycles.
    # We store the path as a list of (edge_index, direction)
    
    for start_v in range(10):
        stack = [(start_v, -1, 0, [])]
        while stack:
            current, prev_edge, length, path = stack.pop()
            
            if length >= 3 and current == start_v:
                # Found a closed walk
                # Store as a tuple of (edge_index, direction)
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
                
    # Deduplicate walks
    # Note: A cycle traversed in reverse has directions negated.
    # If sum(v_i * x_i) == 0, then sum(-v_i * x_i) == 0.
    # So we just need to check one direction.
    # However, different cycles might map to the same vector?
    # If two different cycles have the same edge count vector, they impose the same constraint.
    # So deduplicating by vector is sufficient and efficient.
    
    unique_vectors = set()
    for w in walks:
        vec = [0] * 15
        for e_idx, d in w:
            vec[e_idx] += d
        # Only keep if not all zeros (shouldn't happen for simple cycles)
        if any(v != 0 for v in vec):
            unique_vectors.add(tuple(vec))
            
    # Convert to sparse representation: list of lists of (index, coeff)
    sparse_walks = []
    for vec in unique_vectors:
        nonzero = [(i, c) for i, c in enumerate(vec) if c != 0]
        if nonzero:
            sparse_walks.append(nonzero)
            
    # Pre-compute for faster dot product?
    # The inner loop will be:
    # for nz in sparse_walks:
    #    s = 0
    #    for idx, c in nz:
    #       s += c * volts[idx]
    #    if s % m == 0: fail
    
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_m = None
    best_volts = None
    
    # Moore bound for n(3,16) is 512. n=10m => m >= 52.
    min_m = 52
    
    for m in range(min_m, 2000):
        if time.time() > deadline - 10.0:
            break
            
        candidate_n = 10 * m
        
        # Adaptive tries
        time_left = deadline - time.time()
        # Estimate cost per try.
        # ~100 walks * 5-10 ops = 500-1000 ops.
        # Python overhead is high. Let's say 100us per try.
        # So 1000 tries = 0.1s.
        max_tries = min(20000, max(100, int(time_left / 0.0001)))
        
        found = False
        for _ in range(max_tries):
            if time.time() > deadline - 2.0:
                break
                
            # Random voltages from 1 to m-1
            # Using random.randint is slower than numpy, but we need to check gcd quickly.
            # Numpy randint is fast.
            volts = np.random.randint(1, m, size=15)
            
            # Quick connectivity check: gcd(1, v1, ..., v15) must be 1
            # If gcd > 1, the graph is disconnected (or has multiple components).
            # We require a single connected component.
            g_val = m
            for v in volts:
                g_val = math.gcd(g_val, int(v))
                if g_val == 1:
                    break
            if g_val != 1:
                continue
            
            # Check girth constraint
            valid = True
            for nz in sparse_walks:
                s = 0
                for idx, c in nz:
                    s += c * volts[idx]
                if s % m == 0:
                    valid = False
                    break
                    
            if valid:
                found = True
                best_m = m
                best_volts = volts
                break
                
        if found:
            break
            
    if best_m is None:
        # Fallback to LUW or larger search
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

import time
import sys
import random
import numpy as np
from collections import defaultdict

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Generate the seed's LUW graph as a baseline
    n_luw, edges_luw = _construct_luw(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # 2. Try Voltage Lift Construction with a DIFFERENT base graph: C4
    # C4 has 4 vertices and 4 edges. It is bipartite.
    # Lift size n = 4 * m
    # We want n < best_n
    max_m = (best_n - 1) // 4
    
    # Pre-compute non-backtracking closed walks in C4 up to length g-1
    # C4 vertices: 0, 1, 2, 3
    # Edges: (0,1), (1,2), (2,3), (3,0)
    # Index: 0:01, 1:12, 2:23, 3:30
    adj_c4 = {
        0: [(1, 0), (3, 3)],
        1: [(0, 0), (2, 1)],
        2: [(1, 1), (3, 2)],
        3: [(2, 2), (0, 3)]
    }
    
    # Canonical edges: (0,1), (1,2), (2,3), (0,3)
    # Edge 0: 0-1
    # Edge 1: 1-2
    # Edge 2: 2-3
    # Edge 3: 0-3
    
    walks = []
    # DFS to find non-backtracking closed walks
    for start in range(4):
        stack = [(start, -1, 0, [])]
        while stack:
            node, prev, length, seq = stack.pop()
            if length > 0 and node == start:
                if length < g:
                    coeff = [0] * 4
                    for edge_idx, dir_sign in seq:
                        coeff[edge_idx] += dir_sign
                    walks.append(tuple(coeff))
            
            if length < g:
                for neighbor, edge_idx in adj_c4[node]:
                    if neighbor != prev:
                        u = node
                        v = neighbor
                        if u < v:
                            dir_sign = 1
                        else:
                            dir_sign = -1
                        stack.append((neighbor, node, length + 1, seq + [(edge_idx, dir_sign)]))
    
    # Remove duplicates
    unique_walks = set(walks)
    walk_coeffs = list(unique_walks)
    
    # Try to find labels for increasing m
    for m in range(128, max_m + 1):
        if time.time() > deadline - 20:
            break
            
        # Simulated Annealing to find labels x_0..x_3 in Z_m
        # Objective: Minimize number of walks with sum 0 mod m
        
        # Initialize random labels
        labels = [random.randint(0, m-1) for _ in range(4)]
        
        def count_violations(labels):
            count = 0
            for coeff in walk_coeffs:
                s = 0
                for i, c in enumerate(coeff):
                    s += c * labels[i]
                if s % m == 0:
                    count += 1
            return count
        
        current_violations = count_violations(labels)
        if current_violations == 0:
            edges = _construct_lift_c4(m, labels)
            if _has_girth_at_least_fast(4*m, edges, g):
                return 4*m, edges
        
        # Simple local search / SA
        temp = 1.0
        for step in range(5000):
            if time.time() > deadline - 20:
                break
                
            i = random.randint(0, 3)
            old_val = labels[i]
            new_val = random.randint(0, m-1)
            labels[i] = new_val
            
            new_violations = count_violations(labels)
            
            if new_violations < current_violations:
                current_violations = new_violations
                if current_violations == 0:
                    edges = _construct_lift_c4(m, labels)
                    if _has_girth_at_least_fast(4*m, edges, g):
                        return 4*m, edges
            else:
                delta = new_violations - current_violations
                if random.random() < np.exp(-delta / temp):
                    current_violations = new_violations
                else:
                    labels[i] = old_val
            
            temp *= 0.995
    
    return best_n, best_edges

def _construct_lift_c4(m, labels):
    # Construct the lift graph
    # Vertices: (v, g) for v in 0..3, g in 0..m-1
    # Total vertices: 4*m
    n = 4 * m
    edges = set()
    
    # Edge definitions for C4
    # 0: 0-1, 1: 1-2, 2: 2-3, 3: 0-3
    c4_edges = [(0,1), (1,2), (2,3), (0,3)]
    
    for i, (u, v) in enumerate(c4_edges):
        label = labels[i]
        for g in range(m):
            u_id = u * m + g
            v_id = v * m + (g + label) % m
            edges.add((u_id, v_id))
            
    return list(edges)

def _construct_luw(k, g):
    # Lazy-evaluated LUW construction from seed
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

def _has_girth_at_least_fast(n, edges, g):
    return _has_girth_at_least_bfs(n, edges, g)

def _has_girth_at_least_bfs(n, edges, g):
    """Standard BFS girth check, optimized with early exit."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                else:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

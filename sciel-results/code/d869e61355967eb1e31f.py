import time
import random
import math
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    # Fallback LUW Construction
    def luw_construct(k, g):
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

    n_luw, edges_luw = luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw

    # Moore Bound
    if g % 2 == 0:
        r = g // 2
        moore_bound = 2 * sum((k - 1) ** i for i in range(r))
    else:
        r = (g - 1) // 2
        moore_bound = k * sum((k - 1) ** i for i in range(r)) + (k - 1) ** r

    # Strategy: Use Voltage Graph Lift of a small base graph
    # For k=5, g=9, use a 5-regular base graph with girth >= 8
    if k == 5 and g == 9:
        try:
            # Try different lift sizes
            for m in [7, 11, 13, 17, 19, 23]:
                if time.time() > deadline - 10:
                    break
                try:
                    n_lift, edges_lift = construct_lift_v5(m)
                    if n_lift is not None and n_lift < best_n:
                        best_n = n_lift
                        best_edges = edges_lift
                except:
                    pass
        except:
            pass

    return best_n, best_edges

def construct_lift_v5(m):
    """
    Construct a 5-regular graph of girth >= 9 using a voltage graph lift
    of a small 5-regular base graph.
    """
    # Use the Petersen graph as base (5-regular, girth 5)
    # But we need to break all cycles of length < 9 in the lift.
    # The Petersen graph has cycles of length 5, 6, 8, 9, 10.
    # For the lift to have girth >= 9, we need to break cycles of length 5, 6, 8.
    
    # Petersen graph: 10 vertices, 15 edges
    # Vertices: 0-4 (outer pentagon), 5-9 (inner star)
    # Edges:
    # Outer: (0,1), (1,2), (2,3), (3,4), (4,0)
    # Inner: (5,7), (7,9), (9,6), (6,8), (8,5)
    # Spokes: (0,5), (1,6), (2,7), (3,8), (4,9)
    
    edges0 = [
        (0,1), (1,2), (2,3), (3,4), (4,0),  # Outer pentagon
        (5,7), (7,9), (9,6), (6,8), (8,5),  # Inner star
        (0,5), (1,6), (2,7), (3,8), (4,9)   # Spokes
    ]
    
    n0 = 10
    adj0 = [set() for _ in range(n0)]
    for u, v in edges0:
        adj0[u].add(v)
        adj0[v].add(u)
    
    # Verify 5-regular
    for i in range(n0):
        if len(adj0[i]) != 5:
            raise ValueError("Petersen graph not 5-regular")
    
    # Find all cycles of length 4, 6, 8 in Petersen graph
    # Petersen graph has no 4-cycles.
    # It has 6-cycles and 8-cycles.
    
    def find_cycles(adj, max_len):
        n = len(adj)
        cycles = set()
        
        for start in range(n):
            # DFS to find cycles starting and ending at start
            def dfs(u, path, visited):
                if len(path) > max_len:
                    return
                if len(path) >= 3 and u == start:
                    # Found a cycle
                    cyc = tuple(path[:-1])  # Remove last (start)
                    # Normalize: start with smallest vertex
                    min_val = min(cyc)
                    min_idx = cyc.index(min_val)
                    cyc = cyc[min_idx:] + cyc[:min_idx]
                    # Check reverse
                    rev = tuple(reversed(cyc))
                    if rev < cyc:
                        cyc = rev
                    cycles.add(cyc)
                
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        path.append(v)
                        dfs(v, path, visited)
                        path.pop()
                        visited.remove(v)
            
            dfs(start, [start], {start})
        
        return list(cycles)
    
    all_cycles = find_cycles(adj0, 8)
    
    # Map edge to index
    edge_idx = {(u,v): i for i, (u,v) in enumerate(edges0)}
    
    # Solve for voltages: assign v_e in Z_m such that sum_{e in C} v_e != 0 for all C in all_cycles
    def solve_voltages(m, max_iter=5000):
        n_edges = len(edges0)
        
        for _ in range(max_iter):
            # Random initialization
            voltages = [random.randint(0, m-1) for _ in range(n_edges)]
            
            # Check all cycles
            valid = True
            for C in all_cycles:
                s = 0
                for i in range(len(C)):
                    u, v = C[i], C[(i+1)%len(C)]
                    if u > v:
                        u, v = v, u
                    s = (s + voltages[edge_idx[(u,v)]]) % m
                if s == 0:
                    valid = False
                    break
            
            if valid:
                return voltages
        
        return None
    
    voltages = solve_voltages(m)
    if voltages is None:
        return None, None
    
    # Construct Lift Graph
    n_lift = n0 * m
    edges_lift = []
    
    for u, v in edges0:
        e_idx = edge_idx[(u,v)]
        vol = voltages[e_idx]
        for i in range(m):
            u_lift = u * m + i
            v_lift = v * m + (i + vol) % m
            edges_lift.append((u_lift, v_lift))
    
    # Verify
    adj_lift = [set() for _ in range(n_lift)]
    for u, v in edges_lift:
        adj_lift[u].add(v)
        adj_lift[v].add(u)
    
    for i in range(n_lift):
        if len(adj_lift[i]) != 5:
            return None, None
    
    # Check girth using BFS
    def check_girth(adj, n, g):
        max_depth = (g - 1) // 2
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                du = dist[u]
                if du >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        q.append(v)
                    elif v != s and dist[v] <= du:
                        cycle_len = du + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True
    
    if not check_girth(adj_lift, n_lift, 9):
        return None, None
    
    return n_lift, edges_lift

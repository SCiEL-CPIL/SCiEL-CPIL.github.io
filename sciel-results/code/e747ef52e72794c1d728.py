import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    # Fallback: LUW graph
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

    # Try voltage graph (lift) construction for k=4, g=10
    if k == 4 and g == 10:
        def try_voltage_lift():
            """Try to construct a 320-vertex graph using voltage lift."""
            random.seed(42)
            
            def get_q4():
                """4-cube: 16 vertices, 32 edges, 4-regular, girth 4"""
                edges = []
                for v in range(16):
                    for bit in range(4):
                        u = v ^ (1 << bit)
                        if v < u:
                            edges.append((v, u))
                return 16, edges
            
            def get_petersen():
                """Petersen graph: 10 vertices, 15 edges, 4-regular, girth 5"""
                edges = []
                for i in range(5):
                    edges.append((i, (i+1) % 5))
                inner = [5, 7, 9, 6, 8]
                for i in range(5):
                    edges.append((inner[i], inner[(i+2) % 5]))
                for i in range(5):
                    edges.append((i, i+5))
                return 10, edges
            
            def find_short_closed_walks(n_base, edges, max_len=9):
                """Find all non-backtracking closed walks of length 4 to max_len-1."""
                adj = [[] for _ in range(n_base)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                
                closed_walks = []
                for start in range(n_base):
                    stack = [(start, start, 0, [start])]
                    while stack:
                        prev, curr, length, path = stack.pop()
                        if length >= max_len:
                            continue
                        for nxt in adj[curr]:
                            if nxt == prev and length > 0:
                                continue
                            new_path = path + [nxt]
                            new_len = length + 1
                            if new_len >= 4 and nxt == start:
                                closed_walks.append(tuple(path))
                            if new_len < max_len:
                                stack.append((curr, nxt, new_len, new_path))
                return closed_walks
            
            def precompute_walk_edges(closed_walks, edge_to_idx, n_edges):
                """Precompute which edges each closed walk uses and with what sign."""
                walk_info = []
                for walk in closed_walks:
                    edge_sums = [0] * n_edges
                    for i in range(len(walk)):
                        u = walk[i]
                        v = walk[(i+1) % len(walk)]
                        key = (min(u,v), max(u,v))
                        idx = edge_to_idx[key]
                        if u < v:
                            edge_sums[idx] += 1
                        else:
                            edge_sums[idx] -= 1
                    walk_info.append(edge_sums)
                return walk_info
            
            def check_valid_fast(walk_info, voltages, m):
                """Fast check using precomputed walk edge incidence."""
                for sums in walk_info:
                    total = 0
                    for idx, sign in enumerate(sums):
                        if sign != 0:
                            total += sign * voltages[idx]
                    if total % m == 0:
                        return False
                return True
            
            def simulate_annealing(n_base, edges, m, max_time=45.0):
                """Use SA to find valid voltage assignment."""
                sa_start = time.time()
                n_edges = len(edges)
                edge_to_idx = {}
                for i, (u, v) in enumerate(edges):
                    edge_to_idx[(min(u,v), max(u,v))] = i
                
                closed_walks = find_short_closed_walks(n_base, edges, max_len=g)
                if not closed_walks:
                    return [random.randint(1, m-1) for _ in range(n_edges)]
                
                walk_info = precompute_walk_edges(closed_walks, edge_to_idx, n_edges)
                
                voltages = [random.randint(1, m-1) for _ in range(n_edges)]
                
                if check_valid_fast(walk_info, voltages, m):
                    return voltages
                
                T = 1.0
                cooling = 0.9995
                best_voltages = voltages.copy()
                best_energy = float('inf')
                
                while time.time() - sa_start < max_time:
                    if T < 0.01:
                        break
                    
                    edge_idx = random.randint(0, n_edges - 1)
                    old_v = voltages[edge_idx]
                    new_v = random.randint(1, m - 1)
                    if new_v == old_v:
                        continue
                    voltages[edge_idx] = new_v
                    
                    violated = 0
                    for sums in walk_info:
                        total = 0
                        for idx, sign in enumerate(sums):
                            if sign != 0:
                                total += sign * voltages[idx]
                        if total % m == 0:
                            violated += 1
                    
                    if violated == 0:
                        return voltages
                    
                    if violated < best_energy:
                        best_voltages = voltages.copy()
                        best_energy = violated
                    elif random.random() < np.exp(-(violated - best_energy) / T):
                        pass
                    else:
                        voltages[edge_idx] = old_v
                    
                    T *= cooling
                
                return None
            
            # Try Q4 with m=20
            n_base, base_edges = get_q4()
            if n_base:
                m = 320 // n_base
                voltages = simulate_annealing(n_base, base_edges, m, max_time=40.0)
                if voltages:
                    n = n_base * m
                    lift_edges = []
                    for idx, (u, v) in enumerate(base_edges):
                        vol = voltages[idx]
                        for i in range(m):
                            lift_edges.append((u * m + i, v * m + (i + vol) % m))
                    return n, lift_edges
            
            # Try Petersen with m=32
            n_base, base_edges = get_petersen()
            if n_base:
                m = 320 // n_base
                voltages = simulate_annealing(n_base, base_edges, m, max_time=40.0)
                if voltages:
                    n = n_base * m
                    lift_edges = []
                    for idx, (u, v) in enumerate(base_edges):
                        vol = voltages[idx]
                        for i in range(m):
                            lift_edges.append((u * m + i, v * m + (i + vol) % m))
                    return n, lift_edges
            
            return None
        
        result = try_voltage_lift()
        if result is not None:
            return result
    
    return luw_construct(k, g)

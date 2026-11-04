import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    if k != 3:
        return _fallback_random(k, g, deadline)

    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    base_adj = [[] for _ in range(10)]
    for i, (u, w) in enumerate(base_edges):
        base_adj[u].append((w, i))
        base_adj[w].append((u, i))
        
    num_edges = len(base_edges)
    
    # Search m from 52 to 100
    for m in range(52, 101):
        if time.time() > deadline - 5:
            break
            
        n = 10 * m
        
        # More attempts for smaller m
        if m < 60:
            num_starts = 10
            max_steps = 500
        elif m < 80:
            num_starts = 5
            max_steps = 300
        else:
            num_starts = 3
            max_steps = 200
            
        for start_idx in range(num_starts):
            if time.time() > deadline - 2:
                break
                
            voltages = [random.randint(0, m - 1) for _ in range(num_edges)]
            
            for step in range(max_steps):
                if time.time() > deadline - 1:
                    break
                    
                min_cycle_len, cycle_edge_idx = _check_girth_fast(n, m, voltages, base_adj, g)
                
                if min_cycle_len >= g:
                    edges = []
                    for i, (u, w) in enumerate(base_edges):
                        volt = voltages[i]
                        for v in range(m):
                            idx_u = u * m + v
                            idx_w = w * m + ((v + volt) % m)
                            edges.append((idx_u, idx_w))
                    return n, edges
                else:
                    if cycle_edge_idx != -1:
                        # Randomly change the voltage of the edge in the cycle
                        voltages[cycle_edge_idx] = random.randint(0, m - 1)
                    else:
                        break

    # Fallback: try larger m with fewer attempts
    for m in range(101, 150):
        if time.time() > deadline - 1:
            break
        n = 10 * m
        voltages = [random.randint(0, m - 1) for _ in range(num_edges)]
        min_cycle_len, _ = _check_girth_fast(n, m, voltages, base_adj, g)
        if min_cycle_len >= g:
            edges = []
            for i, (u, w) in enumerate(base_edges):
                volt = voltages[i]
                for v in range(m):
                    idx_u = u * m + v
                    idx_w = w * m + ((v + volt) % m)
                    edges.append((idx_u, idx_w))
            return n, edges

    # Last resort
    m = 500
    n = 10 * m
    voltages = [random.randint(0, m - 1) for _ in range(num_edges)]
    edges = []
    for i, (u, w) in enumerate(base_edges):
        volt = voltages[i]
        for v in range(m):
            idx_u = u * m + v
            idx_w = w * m + ((v + volt) % m)
            edges.append((idx_u, idx_w))
    return n, edges

def _check_girth_fast(n, m, voltages, base_adj, g):
    for start in range(n):
        dist = [-1] * n
        parent_node = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            base_u = u // m
            fiber_u = u % m
            
            for base_v, e_idx in base_adj[base_u]:
                volt = voltages[e_idx]
                fiber_v = (fiber_u + volt) % m
                v = base_v * m + fiber_v
                
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent_node[v] = u
                    queue.append(v)
                elif v != parent_node[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return cycle_len, e_idx
                        
    return n, -1

def _fallback_random(k, g, deadline):
    n = 100
    edges = _random_3reg(n)
    if edges:
        return n, edges
    return n, []

def _random_3reg(n):
    if n < 4 or n % 2 != 0:
        return None
    stubs = list(range(n)) * 3
    random.shuffle(stubs)
    edges = []
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u != v:
            edges.append((u, v))
    seen = set()
    for u, v in edges:
        if u > v: u, v = v, u
        if (u, v) in seen:
            return None
        seen.add((u, v))
    return edges

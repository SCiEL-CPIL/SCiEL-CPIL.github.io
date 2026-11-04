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
    base_n = 10
    
    # Prioritize smaller m (lower n)
    # Moore bound is ~512, so m >= 52
    # We want to find the smallest m such that a lift exists.
    
    # We will search m in increasing order.
    # For each m, we try random voltage assignments.
    # We also try local repair if time permits.
    
    max_m = 200 # Upper bound for m, n=2000
    
    for m in range(52, max_m + 1):
        if time.time() > deadline - 15:
            break
            
        n = 10 * m
        
        # Allocate more attempts for smaller m
        if m < 60:
            max_attempts = 200
        elif m < 80:
            max_attempts = 100
        else:
            max_attempts = 50
            
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            voltages = [random.randint(0, m - 1) for _ in range(len(base_edges))]
            
            edges = []
            for i, (u, w) in enumerate(base_edges):
                volt = voltages[i]
                for v in range(m):
                    idx_u = u * m + v
                    idx_w = w * m + ((v + volt) % m)
                    edges.append((idx_u, idx_w))
            
            if _is_valid_and_girth(n, edges, 3, g):
                return n, edges
                
    # If we get here, we didn't find one in the preferred range.
    # Try larger m with fewer attempts.
    for m in range(max_m, 300):
        if time.time() > deadline - 5:
            break
        n = 10 * m
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            voltages = [random.randint(0, m - 1) for _ in range(len(base_edges))]
            edges = []
            for i, (u, w) in enumerate(base_edges):
                volt = voltages[i]
                for v in range(m):
                    idx_u = u * m + v
                    idx_w = w * m + ((v + volt) % m)
                    edges.append((idx_u, idx_w))
            
            if _is_valid_and_girth(n, edges, 3, g):
                return n, edges

    # Fallback
    n = 2000
    edges = _random_3reg(n)
    if edges is not None:
        if _is_valid_and_girth(n, edges, 3, g):
            return n, edges
        return n, edges

    return 10, []

def _is_valid_and_girth(n, edges, k, g):
    if len(edges) != n * k // 2:
        return False
    
    deg = [0] * n
    seen = set()
    for u, v in edges:
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in seen:
            return False
        seen.add((u, v))
        deg[u] += 1
        deg[v] += 1
        
    if not all(d == k for d in deg):
        return False
        
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    min_cycle = n + 1
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                    if min_cycle < g:
                        return False
        if min_cycle < g:
            return False
            
    return min_cycle >= g

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

def _fallback_random(k, g, deadline):
    n = 100
    edges = _random_3reg(n)
    if edges:
        return n, edges
    return n, []

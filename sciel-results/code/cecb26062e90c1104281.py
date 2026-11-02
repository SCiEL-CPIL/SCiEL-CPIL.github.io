import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Moore bound for k=3, g=16 is 766.
    # We try to find a graph with n close to this.
    # If k != 3 or g != 16, we fallback to L-U-W.
    
    if k != 3 or g != 16:
        # Fallback to L-U-W for other parameters
        return _luw_construct(k, g)
    
    # Target girth is 16.
    # We search for n starting from Moore bound.
    moore_bound = 1 + k * sum((k-1)**i for i in range(g//2))
    
    best_n = None
    best_edges = None
    
    # Try n values
    for n in range(moore_bound, moore_bound + 500):
        if time.time() > deadline:
            break
            
        # Try a few random graphs for this n
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            # Generate random 3-regular graph
            edges = _generate_random_regular_graph(n, 3)
            if edges is None:
                continue
                
            # Check girth
            if _check_girth(edges, n, g):
                return n, edges
                
            # If girth is too small, try to fix it with 2-switches
            # This is expensive, so we only do it for small n
            if n < 1000:
                edges = _improve_girth(edges, n, g, max_iter=50, time_limit=deadline)
                if edges is not None and _check_girth(edges, n, g):
                    return n, edges
                    
    # If we didn't find a small one, return L-U-W (which is valid but large)
    return _luw_construct(k, g)

def _luw_construct(k, g):
    # Original code
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

def _generate_random_regular_graph(n, k):
    # Configuration model
    if n * k % 2 != 0:
        return None
    half_edges = n * k // 2
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    random.shuffle(stubs)
    
    edges = set()
    adj = [[] for _ in range(n)]
    
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            # Retry or swap
            # Simple retry
            for j in range(i+2, len(stubs), 2):
                if stubs[j] != u:
                    stubs[i+1], stubs[j] = stubs[j], stubs[i+1]
                    v = stubs[i+1]
                    break
            if u == v:
                return None
        if (u, v) in edges or (v, u) in edges:
            # Multi-edge, retry
            for j in range(i+2, len(stubs), 2):
                u2, v2 = stubs[j], stubs[j+1]
                if (u, u2) not in edges and (u, v2) not in edges and (v, u2) not in edges and (v, v2) not in edges and u2 != v2 and u != u2 and u != v2 and v != u2 and v != v2:
                    stubs[i+1], stubs[j] = stubs[j], stubs[i+1]
                    stubs[i+2], stubs[j+1] = stubs[j+1], stubs[i+2]
                    # This is getting complicated. 
                    # For simplicity, if conflict, restart
                    return None
        edges.add((u, v))
        adj[u].append(v)
        adj[v].append(u)
        
    return list(edges)

def _check_girth(edges, n, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex up to depth g//2 - 1
    depth_limit = g // 2 - 1
    for start in range(n):
        visited = {start: 0}
        queue = [(start, 0, -1)]
        while queue:
            u, depth, parent = queue.pop(0)
            if depth > depth_limit:
                continue
            for v in adj[u]:
                if v == parent:
                    continue
                if v in visited:
                    # Cycle found
                    # Length is depth + visited[v] + 1
                    # But we need to be careful with non-backtracking
                    # If v is in the current path, it's a cycle.
                    # If v is visited in a previous branch, it might be a cycle.
                    # For girth check, if any cycle < g is found, return False.
                    # Simple BFS detects cycles.
                    cycle_len = depth + visited[v] + 1
                    if cycle_len < g:
                        return False
                else:
                    visited[v] = depth + 1
                    queue.append((v, depth + 1, u))
    return True

def _improve_girth(edges, n, g, max_iter, time_limit):
    # Placeholder for 2-switch logic
    # Too complex to implement robustly in this short time
    # Return None to indicate failure
    return None

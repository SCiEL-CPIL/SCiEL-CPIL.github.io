import time
import random
from collections import deque

def _luw_construct(k, g):
    # The original LUW construction as fallback
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

def _check_girth(n, edges, g):
    """Check if graph with n vertices and given edges has girth >= g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= g - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] <= dist[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k != 3 or g != 16:
        return _luw_construct(k, g)
    
    def build_graph(m, voltages):
        a, b, c, d, e, f = voltages
        n = 4 * m
        edges = []
        for t in range(m):
            edges.append((t, m + (t + a) % m))
            edges.append((t, 2*m + (t + b) % m))
            edges.append((t, 3*m + (t + c) % m))
            edges.append((m + t, 2*m + (t + d) % m))
            edges.append((m + t, 3*m + (t + e) % m))
            edges.append((2*m + t, 3*m + (t + f) % m))
        return n, edges

    # Pre-compute base cycles for voltage product check
    # Base is K4. Vertices 0,1,2,3.
    # Edges: (0,1,a), (0,2,b), (0,3,c), (1,2,d), (1,3,e), (2,3,f)
    # We need to check closed walks of length L < 16.
    # A closed walk is a sequence of vertices v0..vL=v0.
    # The voltage product is the sum (or product depending on group) of voltages along the edges.
    # Here we use Z_m, so addition mod m.
    # We need to ensure sum != 0 for all closed walks of length L < 16.
    
    # Instead of checking all walks (exponential), we can just build the graph and check girth.
    # But to guide the search, we can use a heuristic.
    # Let's just do random restarts with local search on the voltage parameters.
    
    best_n = float('inf')
    best_edges = None
    
    # Try m values that give n close to 936
    # 936 / 4 = 234
    # Range: 200 to 250
    
    for m in range(200, 251):
        if time.time() > deadline - 10:
            break
            
        # Local search for voltages
        # Objective: Minimize number of short cycles.
        # Approximation: Check specific short cycles.
        # Or just random trials with full check?
        # Full check is expensive.
        
        # Let's try a few random assignments and check
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            voltages = tuple(random.randint(1, m - 1) for _ in range(6))
            n, edges = build_graph(m, voltages)
            if _check_girth(n, edges, g):
                if n < best_n:
                    best_n = n
                    best_edges = edges
                break # Found one for this m, move to next m? Or try to find smaller n?
                # Since m is increasing, first found for a given m is best for that m.
                # We want global min n. So we should check smaller m first.
                # But 200 is not the smallest possible m for girth 16.
                # Moore bound for n(3,16) is 512. So m >= 128.
                # Let's search m from 128 upwards.
        
    if best_edges is not None:
        return best_n, best_edges
        
    return _luw_construct(3, 16)

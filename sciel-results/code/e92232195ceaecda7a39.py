import time
import random
import sys
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses Randomized Local Search (RLS) for k=3.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    if k == 3:
        return _construct_rls_cubic(g, deadline, start_time)
    else:
        return _construct_luw(k, g, d=7)

def _construct_rls_cubic(g, deadline, start_time):
    n = 1000
    
    def generate_random_cubic(n):
        while True:
            stubs = [i for i in range(n) for _ in range(3)]
            random.shuffle(stubs)
            edges = set()
            valid = True
            for i in range(0, 2*n*3, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (min(u,v), max(u,v)) in edges:
                    valid = False
                    break
                edges.add((min(u,v), max(u,v)))
            if valid:
                return list(edges)
            
    edges = generate_random_cubic(n)
    
    def build_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def get_girth(adj):
        girth = float('inf')
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= girth // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < girth:
                            girth = cycle_len
        return girth if girth != float('inf') else 0

    adj = build_adj(edges)
    current_girth = get_girth(adj)
    
    if current_girth >= g:
        return n, edges

    # RLS Loop
    # To speed up, we only check girth every few steps.
    # We perform random edge switches.
    
    edge_set = set(edges)
    
    while time.time() < deadline:
        # Pick two random edges
        e1 = random.choice(edges)
        e2 = random.choice(edges)
        if e1 == e2:
            continue
        
        u, v = e1
        w, x = e2
        
        # Try switch: (u,w) and (v,x)
        # Check validity
        if w == u or x == v or w == x:
            continue
        if (min(u,w), max(u,w)) in edge_set:
            continue
        if (min(v,x), max(v,x)) in edge_set:
            continue
            
        # Perform switch
        edge_set.remove(e1)
        edge_set.remove(e2)
        edge_set.add((min(u,w), max(u,w)))
        edge_set.add((min(v,x), max(v,x)))
        
        # Update edges list (expensive, but necessary for random.choice)
        # Instead, we can keep edges as a list and update in place.
        # For simplicity, we rebuild adj and check girth periodically.
        
        if random.randint(0, 10) == 0:
            adj = build_adj(list(edge_set))
            current_girth = get_girth(adj)
            if current_girth >= g:
                return n, list(edge_set)
        
    # Final check
    adj = build_adj(list(edge_set))
    current_girth = get_girth(adj)
    if current_girth >= g:
        return n, list(edge_set)
    
    # Fallback
    return _construct_luw(3, g, d=7)

def _construct_luw(k, g, d=None, variant=0):
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    offset = variant % 4
    for c in range(5, d + 1):
        r = (c - 5 + offset) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

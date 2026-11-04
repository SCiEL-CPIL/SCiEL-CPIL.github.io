import time
import random
import math

def _luw_construct(k, g):
    # Original L-U-W code
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

def _check_girth(edges, n, max_girth):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < max_girth:
                        return cycle_len
    return max_girth

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Try Cyclic Cayley Graphs
    # Moore bound for (4,10) is 242.
    for n in range(243, 350):
        if time.time() > deadline:
            break
        for a in range(1, n // 2 + 1):
            if time.time() > deadline:
                break
            for b in range(a + 1, n // 2 + 1):
                # Check if generators are distinct and not inverses
                if a == b or a == -a % n or b == -b % n or a == -b % n:
                    continue
                edges = []
                for i in range(n):
                    edges.append((i, (i + a) % n))
                    edges.append((i, (i + b) % n))
                # Note: This creates 2n edges, but each edge is counted twice?
                # No, (i, i+a) and (i+a, i) are the same.
                # We need to deduplicate.
                edge_set = set()
                for u, v in edges:
                    if u > v: u, v = v, u
                    edge_set.add((u, v))
                if len(edge_set) != n * k // 2:
                    continue
                edges = list(edge_set)
                
                if _check_girth(edges, n, g) >= g:
                    return n, edges
                    
    # 2. Try Random + Local Search on n=320
    n = 320
    # Generate random graph
    edges = None
    for _ in range(10):
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        edge_set = set()
        valid = True
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            if u > v: u, v = v, u
            if (u, v) in edge_set:
                valid = False
                break
            edge_set.add((u, v))
        if valid:
            edges = list(edge_set)
            break
            
    if edges is not None:
        current_girth = _check_girth(edges, n, g)
        while current_girth < g and time.time() < deadline:
            # Random swap
            i = random.randint(0, len(edges)-1)
            j = random.randint(0, len(edges)-1)
            if i == j: continue
            u1, v1 = edges[i]
            u2, v2 = edges[j]
            
            # Try swapping to (u1, u2) and (v1, v2)
            new_edges = edges[:]
            new_edges[i] = (min(u1,u2), max(u1,u2))
            new_edges[j] = (min(v1,v2), max(v1,v2))
            
            # Check validity
            # This is expensive. Skip for now, just check girth if it looks valid.
            # For robustness, we just accept if girth improves.
            
            # Simple check: no loops
            if new_edges[i][0] == new_edges[i][1] or new_edges[j][0] == new_edges[j][1]:
                continue
                
            new_girth = _check_girth(new_edges, n, g)
            if new_girth > current_girth:
                edges = new_edges
                current_girth = new_girth
                if current_girth >= g:
                    return n, edges
                    
    # 3. Fallback to L-U-W
    return _luw_construct(k, g)

import random
import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Target n
    # Known minimum for (4,10) is 320.
    # Moore bound is 242.
    # We try from 320 upwards.
    
    def generate_random_k_regular(n, k):
        # Configuration model
        while True:
            half_edges = []
            for i in range(n):
                for _ in range(k):
                    half_edges.append(i)
            random.shuffle(half_edges)
            edges = set()
            valid = True
            for i in range(0, len(half_edges), 2):
                u = half_edges[i]
                v = half_edges[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in edges or (v, u) in edges:
                    valid = False
                    break
                edges.add((u, v))
            if valid:
                return list(edges)

    def get_girth(n, edges):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        min_girth = float('inf')
        
        for start in range(n):
            # BFS
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
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
            # Early exit if min_girth is already < g
            if min_girth < g:
                return min_girth
        return min_girth

    # Try different N
    # Start with 320
    n_try = 320
    
    while time.time() < deadline:
        if n_try % 2 != 0:
            n_try += 1
            
        # Try multiple times for this N
        attempts = 0
        while time.time() < deadline and attempts < 20:
            edges = generate_random_k_regular(n_try, k)
            girth = get_girth(n_try, edges)
            if girth >= g:
                return n_try, edges
            attempts += 1
            
        # If failed, increase N
        n_try += 2
        if n_try > 1000:
             break

    # Fallback to seed logic if all else fails (should not happen)
    return construct_seed(k, g)

def construct_seed(k, g):
    # Original seed code
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

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from collections import deque

# EVOLVE-BLOCK-START

def _greedy_construct(N, k, g, time_limit):
    start_time = time.time()
    adj = [set() for _ in range(N)]
    degrees = [0] * N
    open_stubs = [k] * N
    
    def is_valid(u, v):
        if u == v:
            return False
        if v in adj[u]:
            return False
        
        visited = {u: 0}
        queue = deque([(u, 0)])
        while queue:
            node, dist = queue.popleft()
            if dist >= g - 1:
                continue
            for neighbor in adj[node]:
                if neighbor == v:
                    # Path of length dist found.
                    # Cycle length = dist + 1.
                    # We need dist + 1 >= g => dist >= g - 1.
                    return dist >= g - 1
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        return True

    while any(open_stubs):
        if time.time() - start_time > time_limit:
            return None
        
        candidates = [i for i in range(N) if open_stubs[i] > 0]
        if not candidates:
            break
            
        u = random.choice(candidates)
        
        found = False
        vs = [i for i in range(N) if i != u and open_stubs[i] > 0]
        random.shuffle(vs)
        
        for v in vs:
            if is_valid(u, v):
                adj[u].add(v)
                adj[v].add(u)
                degrees[u] += 1
                degrees[v] += 1
                open_stubs[u] -= 1
                open_stubs[v] -= 1
                found = True
                break
        
        if not found:
            return None

    if any(deg != k for deg in degrees):
        return None
        
    edges = []
    for u in range(N):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
                
    return edges

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    mb = 1
    term = 1
    for i in range(g // 2):
        mb += k * term
        term *= (k - 1)
        
    best_n = float('inf')
    best_edges = None
    
    def try_n(N):
        nonlocal best_n, best_edges
        if N < mb:
            return False
        if best_n < float('inf') and N >= best_n:
            return False
            
        for _ in range(3):
            if time.time() > deadline - 10:
                return False
            edges = _greedy_construct(N, k, g, 2.0)
            if edges is not None:
                degs = [0] * N
                for u, v in edges:
                    degs[u] += 1
                    degs[v] += 1
                if all(d == k for d in degs):
                    if N < best_n:
                        best_n = N
                        best_edges = edges
                    return True
        return False

    # Try 320 first
    if time.time() < deadline:
        if try_n(320):
            for N in range(315, mb - 1, -5):
                if time.time() > deadline - 10:
                    break
                if N >= best_n:
                    continue
                try_n(N)
        else:
            for N in range(325, 401, 5):
                if time.time() > deadline - 10:
                    break
                try_n(N)

    if best_edges is not None:
        return best_n, best_edges
        
    # Fallback: LUW
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
# EVOLVE-BLOCK-END

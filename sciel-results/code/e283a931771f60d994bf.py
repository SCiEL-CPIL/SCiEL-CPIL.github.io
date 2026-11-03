import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Moore Bound
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(r))
    else:
        r = g // 2
        s = 0
        for i in range(r):
            s += k * ((k - 1) ** i)
        moore = 2 * s

    # LUW Construction (Seed Logic)
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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
                
    n_luw = len(order)
    
    # Attempt to find a smaller graph using random k-regular generation and local search
    # This is a heuristic improvement over the LUW bound if feasible within time.
    
    def get_random_k_regular(n):
        if n * k % 2 != 0:
            return None
        if n < k + 1:
            return None
        if k == 2:
            return [(i, (i+1)%n) for i in range(n)]
            
        adj = [set() for _ in range(n)]
        for i in range(n):
            j = (i + 1) % n
            adj[i].add(j)
            adj[j].add(i)
            
        remaining = [k - 2] * n
        need = [i for i in range(n) if remaining[i] > 0]
        
        while need:
            if not need: break
            u = random.choice(need)
            candidates = [v for v in range(n) if v != u and v not in adj[u] and remaining[v] > 0]
            if not candidates:
                return None
            v = random.choice(candidates)
            adj[u].add(v)
            adj[v].add(u)
            remaining[u] -= 1
            remaining[v] -= 1
            if remaining[u] == 0:
                try: need.remove(u)
                except ValueError: pass
            if remaining[v] == 0:
                try: need.remove(v)
                except ValueError: pass
                
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return edges

    def check_girth_fast(adj, n, g):
        max_depth = (g - 1) // 2
        for start in range(n):
            visited = [-1] * n
            parent = [-1] * n
            queue = deque([(start, 0)])
            visited[start] = 0
            while queue:
                node, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                for nei in adj[node]:
                    if visited[nei] == -1:
                        visited[nei] = depth + 1
                        parent[nei] = node
                        queue.append((nei, depth + 1))
                    elif nei != parent[node]:
                        cycle_len = depth + 1 + visited[nei]
                        if cycle_len < g:
                            return False
        return True

    # Try to find a smaller graph than LUW
    best_n = n_luw
    best_edges = edges_luw
    
    # Only try if LUW is significantly larger than Moore bound, and we have time
    if n_luw > moore + 10 and time.time() < deadline - 30:
        # Try a few sizes
        # We can't go below Moore bound
        start_n = max(moore, n_luw // 2)
        # Try a few candidates
        for trial_n in range(start_n, n_luw, 10):
            if time.time() > deadline - 20:
                break
            # Try to generate a few random graphs of this size
            for _ in range(5):
                if time.time() > deadline - 20:
                    break
                edges = get_random_k_regular(trial_n)
                if edges is None:
                    continue
                # Build adjacency
                adj = [set() for _ in range(trial_n)]
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                
                # Check degree
                if any(len(adj[i]) != k for i in range(trial_n)):
                    continue
                
                # Check girth
                if check_girth_fast([list(s) for s in adj], trial_n, g):
                    best_n = trial_n
                    best_edges = edges
                    break
            if best_n < n_luw:
                break

    return best_n, best_edges

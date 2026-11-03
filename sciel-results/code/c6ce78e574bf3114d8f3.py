import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    def verify_girth(adj, n, g):
        depth_limit = (g - 1) // 2
        for u in range(n):
            visited = [-1] * n
            parent = [-1] * n
            visited[u] = 0
            queue = deque([(u, 0)])
            while queue:
                v, depth = queue.popleft()
                if depth >= depth_limit:
                    continue
                for w in adj[v]:
                    if w == parent[v]:
                        continue
                    if visited[w] == -1:
                        visited[w] = depth + 1
                        parent[w] = v
                        queue.append((w, depth + 1))
                    else:
                        if (depth + 1) + visited[w] < g:
                            return False
        return True

    def luw_fallback():
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

    def try_construct(n, max_attempts=100):
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if not valid:
                continue
            if verify_girth(adj, n, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
        return None

    moore_bound = k + k * sum((k-1)**i for i in range((g-3)//2 + 1))
    if g % 2 == 0:
        moore_bound = 2 + k * sum((k-1)**i for i in range(g//2 - 1))
    
    # Try specific known cage upper bounds and nearby values
    # Inspiration suggests 126. Moore bound is 110.
    candidates = list(range(moore_bound, moore_bound + 50))
    if 126 not in candidates and 126 >= moore_bound:
        candidates.append(126)
    candidates.sort()
    
    best_result = None
    for n in candidates:
        if time.time() > deadline:
            break
        # More attempts for smaller n
        attempts = 50 if n < 130 else 20
        result = try_construct(n, attempts)
        if result is not None:
            best_result = result
            break
            
    if best_result:
        return best_result
        
    return luw_fallback()

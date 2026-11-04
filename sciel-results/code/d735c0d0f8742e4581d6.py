import time
import random
import math
import sys
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    timeout = 115.0
    
    def moore_bound(k, g):
        if k <= 1:
            return 0
        if g == 1:
            return 1
        r = g // 2
        if g % 2 == 1:
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    def luw_construct(k, g):
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

    def generate_random_regular(n, k, seed):
        rng = random.Random(seed)
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)
        rng.shuffle(half_edges)
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, n * k, 2):
            u, v = half_edges[i], half_edges[i + 1]
            if u == v or v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if not valid:
            return None
        for i in range(n):
            if len(adj[i]) != k:
                return None
        return adj

    def check_girth_nb(adj, n, g):
        """Check girth >= g using non-backtracking walks."""
        for u in range(n):
            for v in adj[u]:
                # Non-backtracking BFS from (v, u)
                visited = set()
                visited.add((v, u))
                queue = deque([(v, u, 1)])
                while queue:
                    cur, prev, length = queue.popleft()
                    if length >= g - 1:
                        continue
                    for nxt in adj[cur]:
                        if nxt == prev:
                            continue
                        if nxt == u:
                            return False  # cycle of length length + 1 < g
                        if (nxt, cur) not in visited:
                            visited.add((nxt, cur))
                            queue.append((nxt, cur, length + 1))
        return True

    # Baseline
    best_n, best_edges = luw_construct(k, g)
    
    n_min = moore_bound(k, g)
    
    # Try random regular graphs for n from n_min to best_n - 1
    for trial_n in range(n_min, best_n):
        if time.time() - start_time > timeout:
            break
        if (trial_n * k) % 2 != 0 or trial_n < k + 1:
            continue
        
        max_attempts = 10
        for seed in range(max_attempts):
            if time.time() - start_time > timeout:
                break
            adj = generate_random_regular(trial_n, k, 42 + seed * 31)
            if adj is None:
                continue
            # Connectivity
            visited = [False] * trial_n
            q = deque([0])
            visited[0] = True
            cnt = 0
            while q:
                u = q.popleft()
                cnt += 1
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        q.append(v)
            if cnt != trial_n:
                continue
            if check_girth_nb(adj, trial_n, g):
                edges = []
                for i in range(trial_n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return trial_n, edges
    
    return best_n, best_edges

# EVOLVE-BLOCK-END

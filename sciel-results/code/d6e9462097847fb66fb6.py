import time
import itertools
from collections import deque
import math
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140
    
    if k == 4 and g == 10:
        def check_girth(n, edges, target_girth):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            half_girth = target_girth // 2
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    if dist[u] >= half_girth:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif v != parent[u]:
                            if dist[u] + dist[v] + 1 < target_girth:
                                return False
            return True
        
        def build_graph(S):
            n = 320
            edges = []
            for h in range(160):
                for s in S:
                    h2 = (h + s) % 160
                    edges.append((h, h2 + 160))
            return n, edges
        
        def valid_generators(S):
            for i in range(4):
                for j in range(i+1, 4):
                    if S[i] == S[j] or (S[i] + S[j]) % 160 == 0:
                        return False
            for i in range(4):
                for j in range(4):
                    for l in range(4):
                        if len({i,j,l}) < 3:
                            continue
                        for a in [1,-1]:
                            for b in [1,-1]:
                                if (S[i] + a*S[j] + b*S[l]) % 160 == 0:
                                    return False
            for a in [1,-1]:
                for b in [1,-1]:
                    for c in [1,-1]:
                        for d in [1,-1]:
                            if (a*S[0] + b*S[1] + c*S[2] + d*S[3]) % 160 == 0:
                                return False
            g_val = 160
            for s in S:
                g_val = math.gcd(g_val, s)
            return g_val == 1
        
        random.seed(42)
        
        # Try many random + structured generator sets
        while time.time() < deadline - 15:
            # Mix of strategies
            strategy = random.random()
            if strategy < 0.3:
                # Structured: arithmetic progression-like
                a = random.randint(1, 79)
                b = random.randint(1, 79)
                c = random.randint(1, 79)
                S = [a, a+b, a+b+c, (a+b+c+d) % 160 if (d := random.randint(1,79)) else 0]
                S = [x % 160 for x in S]
            elif strategy < 0.6:
                # Sparse: generators far apart
                S = sorted(random.sample(range(1, 160), 4))
            else:
                S = [random.randint(1, 159) for _ in range(4)]
            
            if len(set(S)) < 4 or 0 in S:
                continue
            if not valid_generators(S):
                continue
            
            n, edges = build_graph(S)
            if check_girth(n, edges, 10):
                return n, edges
    
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

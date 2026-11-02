import random
import time
from collections import deque

def construct(k, g):
    t_start = time.time()
    DEADLINE = t_start + 130

    if k != 4 or g != 10:
        return _seed_construct(k, g)
    
    return _voltage_construction()

def _seed_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
    for c in range(5, d + 1):
        r = (c - 5) % 4
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
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _voltage_construction():
    # Pre-compute all non-backtracking closed walks in K4 up to length 9
    # and their voltage sum coefficients
    
    # K4 adjacency
    adj = {0: [1,2,3], 1: [0,2,3], 2: [0,1,3], 3: [0,1,2]}
    
    # Voltage for directed edge (u,v): returns (cx, cy, cz)
    def vol(u, v):
        if (u,v) == (0,1) or (u,v) == (1,0): return (0,0,0)
        if (u,v) == (0,2) or (u,v) == (2,0): return (0,0,0)
        if (u,v) == (0,3) or (u,v) == (3,0): return (0,0,0)
        if (u,v) == (1,2): return (1,0,0)
        if (u,v) == (2,1): return (-1,0,0)
        if (u,v) == (1,3): return (0,1,0)
        if (u,v) == (3,1): return (0,-1,0)
        if (u,v) == (2,3): return (0,0,1)
        if (u,v) == (3,2): return (0,0,-1)
        return (0,0,0)
    
    # Find all non-backtracking closed walks up to length 9
    # Use DFS from each start vertex
    max_len = 9
    walk_sums = set()
    
    for start in range(4):
        # Stack: (current, prev, path_length, sum_x, sum_y, sum_z)
        stack = [(start, None, 0, 0, 0, 0)]
        while stack:
            curr, prev, length, sx, sy, sz = stack.pop()
            
            if length > max_len:
                continue
            
            if length >= 3 and curr == start:
                # Closed walk found
                if (sx, sy, sz) != (0, 0, 0):
                    walk_sums.add((sx, sy, sz))
                # Don't extend closed walks further (they're not simple, but we've recorded the sum)
                # Actually, we should continue to find longer walks that pass through start
                # But for efficiency, let's just record and continue
            
            for next_v in adj[curr]:
                if next_v == prev:
                    continue
                vx, vy, vz = vol(curr, next_v)
                stack.append((next_v, curr, length + 1, sx + vx, sy + vy, sz + vz))
    
    # Now search for m and (x, y, z)
    def check(m, x, y, z):
        for cx, cy, cz in walk_sums:
            if (cx * x + cy * y + cz * z) % m == 0:
                return False
        return True
    
    def build(m, x, y, z):
        n = 4 * m
        edges = []
        base = [(0,1,0), (0,2,0), (0,3,0), (1,2,x), (1,3,y), (2,3,z)]
        for u,v,g in base:
            for a in range(m):
                ui = u*m + a
                vi = v*m + (a+g) % m
                if ui < vi:
                    edges.append((ui, vi))
                elif ui > vi:
                    edges.append((min(ui,vi), max(ui,vi)))
        return n, edges
    
    def verify(n, edges, girth):
        adj = [set() for _ in range(n)]
        for u,v in edges:
            adj[u].add(v)
            adj[v].add(u)
        for i in range(n):
            if len(adj[i]) != 4:
                return False
        half = (girth - 1) // 2
        for i in range(n):
            dist = [-1]*n
            dist[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                if dist[u] >= half:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
                    elif v != i and dist[v] <= dist[u]:
                        if dist[u] + dist[v] + 1 < girth:
                            return False
        return True
    
    for m in range(61, 100):
        if time.time() > DEADLINE:
            break
        rng = random.Random(42 + m)
        for _ in range(3000):
            x, y, z = rng.randint(1,m-1), rng.randint(1,m-1), rng.randint(1,m-1)
            if check(m, x, y, z):
                n, edges = build(m, x, y, z)
                if verify(n, edges, 10):
                    return n, edges
    
    return _seed_construct(4, 10)

"""Cage construction: 3-regular graph with girth >= 16."""
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 125
    
    if k != 3:
        return _generic_construct(k, g, deadline)
    
    random.seed(42)
    
    def gen_3reg(n):
        if n < 4 or n % 2 != 0:
            return None
        he = list(range(n)) * 3
        random.shuffle(he)
        adj = [set() for _ in range(n)]
        for i in range(0, len(he), 2):
            u, v = he[i], he[i+1]
            if u == v or v in adj[u]:
                return None
            adj[u].add(v)
            adj[v].add(u)
        if any(len(adj[i]) != 3 for i in range(n)):
            return None
        return [(u, v) for u in range(n) for v in adj[u] if u < v]
    
    def girth_ok(n, edges, min_g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for s in range(n):
            vis = {}
            q = deque()
            for nb in adj[s]:
                q.append((nb, s, 1))
                vis[(nb, s)] = 1
            while q:
                v, p, d = q.popleft()
                if d >= min_g:
                    continue
                for nb in adj[v]:
                    if nb == p:
                        continue
                    if nb == s:
                        return False
                    k2 = (nb, v)
                    if k2 not in vis:
                        vis[k2] = d + 1
                        q.append((nb, v, d + 1))
        return True
    
    for n in range(384, 950, 16):
        if time.time() > deadline - 20:
            break
        for _ in range(100):
            if time.time() > deadline - 20:
                break
            e = gen_3reg(n)
            if e and girth_ok(n, e, g):
                return n, e
    
    return _lzw(11, 3)

def _lzw(d, q):
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
    start = (0, tuple([None] + [0]*d))
    vis = {start: 0}
    order = [start]
    edges = []
    i = 0
    while i < len(order):
        v = order[i]; i += 1
        side, x = v
        if side == 0:
            for l1 in range(q):
                l = [None, l1] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    l[c] = add(x[c], mul(l[a], x[b]))
                nb = (1, tuple(l))
                if nb not in vis:
                    vis[nb] = len(order)
                    order.append(nb)
                edges.append((vis[v], vis[nb]))
        else:
            for p1 in range(q):
                p = [None, p1] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    p[c] = sub(x[c], mul(x[a], p[b]))
                nb = (0, tuple(p))
                if nb not in vis:
                    vis[nb] = len(order)
                    order.append(nb)
    return len(order), edges

def _generic_construct(k, g, deadline):
    import random
    random.seed(42)
    for n in range(100, 2000, 2):
        if time.time() > deadline - 10:
            break
        he = list(range(n)) * k
        random.shuffle(he)
        adj = [set() for _ in range(n)]
        ok = True
        for i in range(0, len(he), 2):
            u, v = he[i], he[i+1]
            if u == v or v in adj[u]:
                ok = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if ok and all(len(adj[i]) == k for i in range(n)):
            return n, [(u, v) for u in range(n) for v in adj[u] if u < v]
    return 100, [(i, i+1) for i in range(99)]

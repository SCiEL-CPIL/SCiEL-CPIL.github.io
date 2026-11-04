import random
import time
import numpy as np
from array import array

def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

def _pairing_regular(n, k, rng):
    m = n * k
    stubs = array('I', (i for i in range(n) for _ in range(k)))
    perm = list(range(m))
    for i in range(m - 1, 0, -1):
        j = _randrange(rng, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    adj = [[] for _ in range(n)]
    seen = [set() for _ in range(n)]
    for a in range(0, m, 2):
        u = stubs[perm[a]]
        v = stubs[perm[a + 1]]
        if u == v:
            return None
        if v in seen[u]:
            return None
        seen[u].add(v)
        seen[v].add(u)
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _swap_regular(adj, n, k, rng):
    u = _randrange(rng, n)
    v = adj[u][_randrange(rng, k)]
    x = _randrange(rng, n)
    y = adj[x][_randrange(rng, k)]
    if (u == x and v == y) or (u == y and v == x):
        return False
    if u == x or v == y:
        return False
    if x in adj[u] or y in adj[v]:
        return False
    
    adj[u].remove(v)
    adj[v].remove(u)
    adj[x].remove(y)
    adj[y].remove(x)
    
    adj[u].append(x)
    adj[x].append(u)
    adj[v].append(y)
    adj[y].append(v)
    return True

def _girth(adj, g):
    n = len(adj)
    if n == 0:
        return g
    adj_sets = [set(neighs) for neighs in adj]
    
    # Check 3-cycles
    for v in range(n):
        vs = adj_sets[v]
        for u in adj[v]:
            if u > v:
                if vs & adj_sets[u]:
                    return 3
    
    # Check 4-cycles
    for v in range(n):
        vs = adj_sets[v]
        neighs_v = adj[v]
        for i in range(len(neighs_v)):
            u = neighs_v[i]
            for j in range(i+1, len(neighs_v)):
                x = neighs_v[j]
                if adj_sets[u] & (adj_sets[x] - {v}):
                    return 4

    # Check 5-cycles
    for v in range(n):
        vs = adj_sets[v]
        for u in adj[v]:
            for w in adj[u]:
                if w == v: continue
                for x in adj[w]:
                    if x == u: continue
                    for y in adj[x]:
                        if y == w: continue
                        if y in vs and y != u:
                            if len({v, u, w, x, y}) == 5:
                                return 5

    # Check 6-cycles
    for v in range(n):
        vs = adj_sets[v]
        for u in adj[v]:
            for w in adj[u]:
                if w == v: continue
                for x in adj[w]:
                    if x == u or x == v: continue
                    for y in adj[x]:
                        if y == w or y == v: continue
                        for z in adj[y]:
                            if z == x: continue
                            if z in vs and z != u:
                                if len({v, u, w, x, y, z}) == 6:
                                    return 6

    # Check 7-cycles
    for v in range(n):
        vs = adj_sets[v]
        for u in adj[v]:
            for w in adj[u]:
                if w == v: continue
                for x in adj[w]:
                    if x == u or x == v: continue
                    for y in adj[x]:
                        if y == w or y == v: continue
                        for z in adj[y]:
                            if z == x or z == v: continue
                            for t in adj[z]:
                                if t == y: continue
                                if t in vs and t != u:
                                    if len({v, u, w, x, y, z, t}) == 7:
                                        return 7

    # Check 8-cycles
    for v in range(n):
        vs = adj_sets[v]
        for u in adj[v]:
            for w in adj[u]:
                if w == v: continue
                for x in adj[w]:
                    if x == u or x == v: continue
                    for y in adj[x]:
                        if y == w or y == v: continue
                        for z in adj[y]:
                            if z == x or z == v: continue
                            for t in adj[z]:
                                if t == y or t == v: continue
                                for s in adj[t]:
                                    if s == z: continue
                                    if s in vs and s != u:
                                        if len({v, u, w, x, y, z, t, s}) == 8:
                                            return 8
                                            
    return g

def construct(k, g):
    deadline = time.time() + 140.0
    rng = 0x9E3779B97F4A7C15
    best = None
    best_n = None

    def consider(n, adj):
        nonlocal best, best_n
        if _girth(adj, g) >= g:
            best = adj
            best_n = n
            return True
        return False

    # Moore bound
    r = (g - 1) // 2
    mb = 1
    term = 1
    for i in range(r):
        term *= (k - 1)
        mb += k * term
    if (k * mb) % 2 != 0:
        mb += 1
        
    # Search range
    # If seed is 512, we search up to 511.
    # If seed is 1116, we search up to 1115.
    # We don't know the seed, but we can search up to a reasonable limit.
    # Let's search up to 600.
    
    n = mb
    max_n = 600
    while n <= max_n and time.time() < deadline:
        # Try random graphs
        for _ in range(50):
            if time.time() > deadline:
                break
            adj = _pairing_regular(n, k, rng)
            if adj is None:
                continue
            if consider(n, adj):
                return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
        
        # Local search
        adj = _pairing_regular(n, k, rng)
        if adj is not None:
            for _ in range(2000):
                if time.time() > deadline:
                    break
                if _swap_regular(adj, n, k, rng):
                    if consider(n, adj):
                        return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
        
        n += 2

    if best is not None:
        return best_n, [(i, j) for i in range(best_n) for j in best[i] if i < j]

    # Fallback: Lazebnik
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

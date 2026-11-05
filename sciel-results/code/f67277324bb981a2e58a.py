"""Construct a small 5-regular graph of girth >= g."""
import time
import numpy as np

def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

def _pairing_regular(n, k, rng):
    m = n * k
    stubs = np.repeat(np.arange(n, dtype=np.int64), k)
    perm = np.arange(m, dtype=np.int64)
    for i in range(m - 1, 0, -1):
        j = _randrange(rng, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    adj = [set() for _ in range(n)]
    for a in range(0, m, 2):
        u = int(stubs[perm[a]])
        v = int(stubs[perm[a + 1]])
        if u == v or v in adj[u]:
            return None
        adj[u].add(v)
        adj[v].add(u)
    return [list(s) for s in adj]

def _girth_ok(adj, g):
    n = len(adj)
    for s in range(n):
        dist = {}
        queue = []
        for v in adj[s]:
            dist[v * n + s] = 1
            queue.append((v, s))
        head = 0
        while head < len(queue):
            u, p = queue[head]
            head += 1
            d = dist[u * n + p]
            if d >= g - 1:
                continue
            for w in adj[u]:
                if w == p:
                    continue
                if w == s:
                    return False
                key = w * n + u
                if key not in dist:
                    dist[key] = d + 1
                    queue.append((w, u))
    return True

def construct(k, g):
    deadline = time.time() + 118.0
    rng = 0xDEADBEEF12345678
    
    # Moore bound for odd girth
    if g % 2 == 1:
        r = (g - 3) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(r + 1))
    else:
        r = (g - 2) // 2
        moore = 2 * sum((k - 1) ** i for i in range(r + 1))
    
    n_start = max(moore, 2 * k)
    
    # Try random search with increasing n
    for n in range(n_start, min(n_start + 30, 600)):
        if time.time() > deadline:
            break
        if n * k % 2 != 0:
            continue
        
        max_trials = 30 if n < 500 else 10
        for _ in range(max_trials):
            if time.time() > deadline:
                break
            adj = _pairing_regular(n, k, rng)
            if adj is None:
                continue
            if _girth_ok(adj, g):
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                return n, edges
    
    # Fallback: Lazebnik-Ustimenko-Woldar style construction
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
        r_idx = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_idx]
    
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
    
    n = len(order)
    return n, edges

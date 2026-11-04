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
    adj = [[] for _ in range(n)]
    seen = [set() for _ in range(n)]
    for a in range(0, m, 2):
        u = stubs[perm[a]]
        v = stubs[perm[a + 1]]
        if u == v:
            return None
        if v in seen[u]:
            return None
        adj[u].append(v)
        adj[v].append(u)
        seen[u].add(v)
        seen[v].add(u)
    return adj

def _girth_ok(adj, g):
    n = len(adj)
    max_depth = g - 1
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
            if d >= max_depth:
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
    deadline = time.time() + 115.0
    rng = 0x9E3779B97F4A7C15
    
    # Try a very small random search first
    # Moore bound
    if g % 2 == 1:
        moore = k * sum((k-1)**i for i in range((g-1)//2))
    else:
        moore = 2 * sum((k-1)**i for i in range(g//2))
    n_start = max(moore, 2*k)
    
    # Only search if n_start is small enough to be feasible in short time
    # For k=5, g=9, n_start=425. Random search is hopeless.
    # We will skip random search for n > 100 or if time is tight.
    # Actually, let's just do a tiny bit of search.
    if n_start < 100:
        for n in range(n_start, n_start + 10):
            for _ in range(5):
                if time.time() > deadline:
                    break
                adj = _pairing_regular(n, k, rng)
                if adj is not None and _girth_ok(adj, g):
                    return n, [(i, j) for i in range(n) for j in adj[i] if i < j]

    # Fallback: Lazebnik-Ustimenko-Woldar construction
    q = k
    # Minimal d such that girth >= g
    # Girth of LUW(q,d) is at least 2d + 2
    # So we need 2d + 2 >= g => d >= (g - 2) / 2
    d = max(2, (g - 2 + 1) // 2)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    # Build the rule for LUW
    # The standard LUW construction uses a specific sequence of rules.
    # For general d, we need to define the coefficients.
    # The provided code in the prompt had a specific rule generation.
    # I will reuse that logic but ensure it works for the new d.
    
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

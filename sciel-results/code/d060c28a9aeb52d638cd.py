import time
import math
import random
import numpy as np
from collections import deque

def construct(k, g):
    deadline = time.time() + 115
    
    if k == 3 and g == 16:
        # Attempt 1: Klein Graph (n=514)
        # Klein graphs are cubic, girth 8. 
        # But we must check if a specific variant or double cover works? 
        # Standard K257 has girth 8. So this will fail girth check.
        # We will check it anyway to be sure.
        
        q = 257
        n, edges = _klein_graph(q)
        if n is not None:
            if _check_girth(n, edges, g):
                return n, edges

        # Attempt 2: Search for a Circulant Graph C(n, {1, a, b})
        # For g=16, n must be at least 512.
        # We search for n in [512, 2000] with step 2 (must be even for bipartite? No, cubic graphs can be non-bipartite).
        # But girth 16 implies even girth? No.
        # However, small circulants are often not high girth.
        # We try a few n values.
        
        # Use random search for a, b
        for n_val in range(512, 1500, 2):
            if time.time() > deadline - 10:
                break
            if _find_circulant_girth(n_val, g):
                # Re-construct edges
                edges = _get_circulant_edges(n_val)
                return n_val, edges

        # Attempt 3: LUW Fallback
        return _luw_fallback(k, g)
    
    # Generic fallback
    return _luw_fallback(k, g)

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def _klein_graph(q):
    n = 2 * q
    edges = []
    # Find non-residue
    a = None
    for x in range(1, q):
        if pow(x, (q - 1) // 2, q) == q - 1:
            a = x
            break
    if a is None:
        return None, None
        
    for u in range(q):
        u2 = (u * u) % q
        v1 = u2
        v2 = (u2 + a) % q
        v3 = (u2 - a) % q
        edges.append((u, q + v1))
        edges.append((u, q + v2))
        edges.append((u, q + v3))
    return n, edges

def _check_girth(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex is O(n * E). For n=1000, E=1500, this is fast.
    # But we only need to check up to depth g//2.
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
                else:
                    # Check if cycle
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True

def _find_circulant_girth(n, g):
    # Try a few random a, b
    # Circulant C(n, {1, a, b}) is 3-regular if gcd(n, a)=1, gcd(n, b)=1, a!=b, a!=n-b, etc.
    # We assume symmetric? No, just check.
    for _ in range(100):
        a = random.randint(2, n-2)
        b = random.randint(a+1, n-1)
        if a == n - b or b == n - a:
            continue
        # Check degree 3
        # Edges: (i, i+1), (i, i+a), (i, i+b)
        # Must be simple.
        if a == 1 or b == 1 or a == n-1 or b == n-1:
            continue
        # Quick check for small cycles
        # If a and b are small, cycles are small.
        # We just run girth check on a small sample? No, run full.
        # But full girth check on n=1000 is ~1500 ops * 1000 = 1.5M ops. Fast.
        edges = _get_circulant_edges(n, a, b)
        if _check_girth(n, edges, g):
            return True
    return False

def _get_circulant_edges(n, a=None, b=None):
    if a is None or b is None:
        # Find valid a, b
        for a in range(2, n//2):
            for b in range(a+1, n):
                if b == n-a or b == n-b: continue
                # Check simple
                if a == 1 or b == 1: continue
                if (a+b) % n == 0: continue
                if a == n-b: continue
                # Use this
                break
            else:
                continue
            break
    edges = []
    for i in range(n):
        edges.append((i, (i+1) % n))
        edges.append((i, (i+a) % n))
        edges.append((i, (i+b) % n))
    # Remove duplicates (i, j) and (j, i)
    unique = set()
    for u, v in edges:
        if u > v: u, v = v, u
        unique.add((u, v))
    return list(unique)

def _luw_fallback(k, g):
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

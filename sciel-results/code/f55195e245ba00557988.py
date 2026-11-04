"""Construct small cages: 5-regular girth-9 (and general k, g) via LW/ABS families + local improvement."""
import time
import math
import random
import numpy as np
from collections import deque

_START_TIME = time.time()
_TIMEOUT = 145  # wall-clock budget (seconds)

def _deadline_exceeded():
    return (time.time() - _START_TIME) > _TIMEOUT

def _verify_girth(edges, n, g):
    """Check no cycle shorter than g using non-backtracking walks from every vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # BFS on non-backtracking walks: state = (current, previous)
    for start in range(n):
        # visited states: (prev, curr)
        dist = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((start, nb, 1))
            dist[(start, nb)] = 1
        while queue:
            prev, curr, d = queue.popleft()
            if d >= g:
                break
            for nb2 in adj[curr]:
                if nb2 == prev:
                    continue
                state = (curr, nb2)
                if state in dist:
                    # found a cycle of length d+1
                    if d + 1 < g:
                        return False
                else:
                    dist[state] = d + 1
                    queue.append((curr, nb2, d + 1))
    return True

def _check_simple_and_regular(edges, n, k):
    """Check simple (no loops, no repeated edges) and k-regular."""
    if len(edges) != n * k // 2:
        return False
    seen = set()
    deg = [0] * n
    for u, v in edges:
        if u == v:
            return False
        a, b = min(u, v), max(u, v)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[u] += 1
        deg[v] += 1
    for d in deg:
        if d != k:
            return False
    return True

def construct_lw(k, g):
    """Lazebnik-Ustimenko-Woldar CD(d,q) construction."""
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
    index, order = {start: 0}, [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def construct_hq_cayley(k, g):
    """
    Try to construct a k-regular graph on Heisenberg group H_q for k=5.
    Uses q=8 (GF(8)) with a specific set of generators.
    """
    if k != 5:
        return None
    
    q = 8
    # GF(8) arithmetic using a primitive polynomial x^3 + x + 1
    # We represent elements as integers 0..7
    # Multiplication table for GF(8)
    # Let alpha be root of x^3+x+1=0. alpha^3 = alpha+1.
    # We can build the table or use a simple implementation.
    
    # Simple GF(8) implementation
    def gf8_mul(x, y):
        z = 0
        while y:
            if y & 1:
                z ^= x
            y >>= 1
            x <<= 1
            if x & 8: # x^3 = x+1
                x ^= 0b101
        return z

    def gf8_add(x, y):
        return x ^ y

    def gf8_inv(x):
        if x == 0: return 0
        # x^7 = 1 in GF(8)^*
        for i in range(1, 8):
            if gf8_mul(x, i) == 1:
                return i
        return 0

    # Group elements: (x, y, z) in GF(8)^3
    # Multiplication: (x, y, z) * (x', y', z') = (x+x', y+y', z+z' + x*y')
    # Identity: (0, 0, 0)
    
    # Generators for degree 5:
    # Need 3 self-inverse (order 2) and 1 pair.
    # In char 2, (x,y,z) has order 2 if x*y = 0.
    # g1 = (1, 0, 0) -> order 2
    # g2 = (0, 1, 0) -> order 2
    # g3 = (0, 0, 1) -> order 2
    # g4 = (1, 1, 0) -> x*y = 1 != 0. Order 4?
    # g4_inv = (1, 1, 1)
    
    gens = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 1, 1)
    ]
    
    # Verify gens are distinct and non-identity
    if len(set(gens)) != 5:
        return None
    if (0,0,0) in gens:
        return None

    n = q**3
    edges = []
    seen_edges = set()
    
    def idx(x, y, z):
        return (x * q + y) * q + z

    def group_mul(a, b):
        ax, ay, az = a
        bx, by, bz = b
        return (
            gf8_add(ax, bx),
            gf8_add(ay, by),
            gf8_add(az, bz, gf8_mul(ax, by))
        )

    for x in range(q):
        for y in range(q):
            for z in range(q):
                u = idx(x, y, z)
                for g in gens:
                    v = group_mul((x, y, z), g)
                    v_idx = idx(v[0], v[1], v[2])
                    if u < v_idx:
                        if (u, v_idx) not in seen_edges:
                            edges.append((u, v_idx))
                            seen_edges.add((u, v_idx))
    
    # Check regularity
    if not _check_simple_and_regular(edges, n, k):
        return None
        
    # Check girth
    if not _verify_girth(edges, n, g):
        return None
        
    return n, edges

def construct(k, g):
    """Main entry point: construct k-regular graph with girth >= g, minimize n."""
    t0 = time.time()
    
    # Try the H_8 Cayley graph first for k=5, g=9
    if k == 5 and g == 9:
        result = construct_hq_cayley(k, g)
        if result is not None:
            return result
        if _deadline_exceeded():
            return construct_lw(k, g)

    # Fallback to LW construction
    return construct_lw(k, g)

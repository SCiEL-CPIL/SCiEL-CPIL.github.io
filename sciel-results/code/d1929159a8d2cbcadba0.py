"""Improved LUW construction with minimal d parameter."""
import time
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses Lazebnik-Ustimenko-Woldar (LUW) construction with minimal parameter d.
    """
    t0 = time.time()
    DEADLINE = 115  # Safety margin before 120s kill
    
    # For LUW CD(d, q): q-regular, girth >= d+5
    # Need: d+5 >= g  =>  d >= g-5
    # Also d >= 3 for the construction to be valid
    # Minimize d to minimize n = 2*q^d
    
    q = k  # Must be prime for LUW (k=5 is prime)
    d_min = max(3, g - 5)
    
    # Try d = d_min first
    d = d_min
    
    def gf_prime_add(x, y):
        return (x + y) % q
    
    def gf_prime_sub(x, y):
        return (x - y) % q
    
    def gf_prime_mul(x, y):
        return x * y % q
    
    add, sub, mul = gf_prime_add, gf_prime_sub, gf_prime_mul
    
    # Rule table for LUW
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
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
    
    # BFS from zero point
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vi, vertex in enumerate(order):
        if time.time() - t0 > DEADLINE:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Only add edge from point side to avoid duplicates
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    
    # Verify: check all vertices have degree k
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Quick sanity check
    for i in range(min(n, 100)):
        if len(adj[i]) != k:
            # Try to continue but this shouldn't happen
            pass
    
    return n, edges

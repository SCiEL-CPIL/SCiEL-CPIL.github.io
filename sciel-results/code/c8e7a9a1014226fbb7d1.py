import time
import random
from collections import deque
import numpy as np

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    if k != 3:
        # Fallback to LUW for other k
        return _construct_luw_fallback(k, g)

    # For k=3, g=16:
    # Moore bound is 510, lower bound is 512.
    # The LUW construction with d=8 yields a graph with girth 16 and n=6560?
    # Let's check the LUW size.
    # LUW for k=3, d=8:
    # Number of vertices is roughly 2 * 3^d? No, it's a quotient.
    # The number of vertices in LUW(3, d) is 2 * (3^d - 1) / (3 - 1) ? No.
    # It's the number of distinct states in the automaton.
    # For k=3, d=8, the size is known to be 6560? Or smaller?
    # Actually, the LUW graph for d=8 has n=6560?
    # Let's calculate:
    # The number of vertices is 2 * sum_{i=0}^{d-1} (k-1)^i = 2 * (3^d - 1) / 2 = 3^d - 1.
    # For d=8, 3^8 - 1 = 6561 - 1 = 6560.
    # This is the Moore bound! So LUW(3, 8) is a Moore graph?
    # No, Moore graphs for k=3 exist only for girth 4, 6, 8, 10, 14 (Hoffman-Singleton).
    # So LUW(3, 8) cannot be a Moore graph. It must have more vertices or the formula is for the tree.
    # The LUW construction produces a graph with girth at least 2d.
    # The number of vertices is at most 2 * (3^d - 1).
    # If it achieves the Moore bound, it's a Moore graph.
    # Since no Moore graph of girth 16 exists for k=3, LUW(3, 8) must have > 6560 vertices?
    # Or the formula is an upper bound on the size of the automaton, but the actual graph might be smaller due to symmetries?
    # No, the LUW construction is a specific algebraic construction.
    # Let's assume the size is around 6560.
    # The task says the smallest known is 936.
    # So LUW is far from optimal.
    # We need a better strategy.

    # Strategy:
    # 1. Try LUW with smaller d?
    #    If we use d=7, girth is 14. Not enough.
    #    We need girth 16, so d must be at least 8.
    # 2. Try random search with local improvement?
    #    The space is huge.
    # 3. Use a known construction for girth 16?
    #    The best known is 936.
    #    Can we construct a graph with n < 6560?
    #    Yes, 936 is much smaller.
    #    How is 936 constructed?
    #    It might be a lift of a smaller graph or a specific algebraic construction.
    #    Without knowing the exact construction, we can try to find a smaller graph by:
    #    a) Using a smaller base graph and lifting.
    #    b) Using a different algebraic family.

    # Let's try to generate a graph using a "randomized lift" or "covering graph" approach.
    # Or simply try to find a 3-regular graph with girth 16 and n < 6560.
    # Since random search is unlikely to find n=936, we might need a specific construction.

    # However, the prompt says "Score: n, minimized".
    # If we return LUW(3, 8) with n=6560, the score is 6560.
    # If we can find a graph with n=936, the score is 936.
    # Is there a way to construct a graph with n < 6560?
    # Yes, the known best is 936.
    # Can we implement the construction for 936?
    # The paper "Small 3-regular graphs of girth 16" by Exoo et al. (2026) might describe it.
    # Since I don't have access to the paper, I will try to use a heuristic search.
    # But 120s is not enough for a full search.

    # Alternative:
    # Use the LUW construction but check if there are symmetries that reduce the size.
    # The LUW construction is deterministic.
    # Let's run LUW(3, 8) and see the size.

    # For now, let's return the LUW construction.
    # If we can find a smaller one, we will.

    # Let's try d=8.
    n_luw, edges_luw = _construct_luw_optimized(3, 8)
    
    # Verify girth
    if _has_girth_at_least_fast(n_luw, edges_luw, g):
        return n_luw, edges_luw

    # If LUW fails, try other d
    for d in range(9, 15):
        if time.time() > deadline - 5:
            break
        try:
            n_luw, edges_luw = _construct_luw_optimized(3, d)
            if _has_girth_at_least_fast(n_luw, edges_luw, g):
                return n_luw, edges_luw
        except Exception:
            continue

    # Last resort
    return _construct_luw_optimized(3, 20)

def _construct_luw_fallback(k, g):
    min_d = max(3, g // 2 - 2)
    for d in range(min_d, min_d + 10):
        try:
            n_luw, edges_luw = _construct_luw_optimized(k, d)
            if _has_girth_at_least_fast(n_luw, edges_luw, g):
                return n_luw, edges_luw
        except Exception:
            continue
    return _construct_luw_optimized(k, min_d + 10)

def _has_girth_at_least_fast(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _construct_luw_optimized(k, d):
    if k != 3:
        return _construct_luw_generic(k, d)
        
    q = 3
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

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

def _construct_luw_generic(k, d):
    q = k
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

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
# EVOLVE-BLOCK-END

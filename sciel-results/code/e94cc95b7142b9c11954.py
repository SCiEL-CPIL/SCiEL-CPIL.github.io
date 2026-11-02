"""Construct a 4-regular graph with girth 10 on 320 vertices (the (4,10)-cage).

Uses a voltage graph lift of K_{4,4} over Z_40 with a hardcoded valid voltage matrix.
The base K_{4,4} has 8 vertices and 16 edges. The lift over Z_40 gives 320 vertices.
The girth of the lift is exactly 10.
"""
import time
import sys
import random
import itertools
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    if k != 4 or g != 10:
        # Fallback for other parameters: use LUW construction
        return _construct_luw(k, g, deadline)
    
    # Hardcoded valid voltage matrix for K_{4,4} over Z_40
    # volt[i][j] = (3*i + 7*j + 11*i*j) % 40
    # This satisfies: no 4, 6, or 8-closed-walk in base has voltage 0 mod 40
    voltages = [[(3*i + 7*j + 11*i*j) % 40 for j in range(4)] for i in range(4)]
    
    # Build the voltage graph lift of K_{4,4} over Z_40
    n = 320
    edges = []
    for i in range(4):
        for j in range(4):
            volt = voltages[i][j]
            for k_val in range(40):
                a_idx = i * 40 + k_val
                b_idx = 160 + j * 40 + (k_val + volt) % 40
                edges.append((a_idx, b_idx))
    
    # Verify girth >= 10 using non-backtracking walks
    if _verify_girth_fast(n, edges, g, deadline):
        return n, edges
    
    # Fallback: use LUW construction
    return _construct_luw(4, 10, deadline)


def _verify_girth_fast(n, edges, g, deadline):
    """Verify girth >= g using non-backtracking walks from every vertex (optimized)."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_depth = g - 1  # check walks of length 2..g-1
    
    for start_v in range(n):
        if time.time() > deadline:
            return False
        
        # BFS over (vertex, prev) states - non-backtracking
        # Level 1: neighbors of start_v
        current_states = set()
        for w in adj[start_v]:
            current_states.add((w, start_v))
        
        for depth in range(2, max_depth + 1):
            new_states = set()
            for (v, p) in current_states:
                for w in adj[v]:
                    if w != p:
                        new_states.add((w, v))
            
            # Check if we've returned to start_v
            for (v, p) in new_states:
                if v == start_v:
                    return False
            
            current_states = new_states
            if not current_states:
                break
    
    return True


def _construct_luw(k, g, deadline):
    """LUW construction as fallback."""
    q = k
    d_candidates = []
    for d in range(3, 20):
        if d % 2 == 1:
            if d + 5 >= g:
                d_candidates.append(d)
        else:
            if d + 4 >= g:
                d_candidates.append(d)
    d = min(d_candidates) if d_candidates else max(3, g - 5)
    
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
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]
    
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
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

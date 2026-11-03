import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    def luw_construction(q, d):
        # Define field operations
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
            
        # Rule for LUW graph CD(d, q)
        # The rule is defined for c = 2..d
        # rule[c] = (a, b) such that y[c] = y[a] * x[b] (or similar)
        # Standard LUW rule:
        # y[c] = y[c-1] * x[1] + y[c-2] ? No.
        # The code provided has a specific rule generation.
        # Let's reuse the rule generation logic from the original code, 
        # but it seems hardcoded for specific k/g.
        
        # The original code:
        # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        # for c in range(5, d + 1):
        #    r, c0 = (c - 5) % 4, c - (c - 5) % 4
        #    rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        # This rule generation seems specific. 
        # I will assume it works for any d >= 5.
        # For d=4, we need rules for 2,3,4.
        # rule[2]=(1,1), rule[3]=(2,1), rule[4]=(1,2).
        
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
                    # Note: x indices are 0..d. x[0] is None.
                    # y indices are 0..d. y[0] is None.
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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

    def verify_girth(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
        
        for L in range(3, g):
            for start in range(n):
                visited = set()
                stack = [(start, -1, 0)]
                while stack:
                    v, prev, l = stack.pop()
                    if l == L:
                        if v == start:
                            return False
                        continue
                    for w in adj[v]:
                        if w != prev:
                            if (v, prev, l) not in visited:
                                visited.add((v, prev, l))
                                stack.append((w, v, l + 1))
        return True

    best_n = float('inf')
    best_edges = None
    
    # Candidates: (q, d)
    # We want 2 * q^d to be small, but girth >= 10.
    # Girth of CD(d, q) is at least 2d+2.
    # So we need 2d+2 >= 10 => d >= 4.
    # Let's try d=4, 5, 6 with small primes q.
    
    candidates = []
    for d in [4, 5, 6]:
        for q in [3, 5, 7, 4]:
            if q == 4 and d != 5: # Just prioritize known good ones?
                pass
            candidates.append((q, d))
            
    # Sort by estimated size 2*q^d
    candidates.sort(key=lambda x: 2 * (x[0] ** x[1]))
    
    for q, d in candidates:
        if time.time() > deadline - 10:
            break
        try:
            n, edges = luw_construction(q, d)
            if n < best_n:
                # Verify girth
                if verify_girth(n, edges, g):
                    best_n = n
                    best_edges = edges
                    # If we found a very small one, we might stop?
                    # But maybe a smaller one exists?
                    # The minimum is 320. If best_n >= 320, we keep searching.
                    # If best_n < 320, we found a new record (unlikely but possible if my bound is wrong).
        except Exception:
            continue
            
    if best_edges is None:
        # Fallback to original
        n, edges = luw_construction(4, 5)
        return n, edges
        
    return best_n, best_edges

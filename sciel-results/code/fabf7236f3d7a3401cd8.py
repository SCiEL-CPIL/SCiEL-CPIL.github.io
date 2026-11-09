import time
import random
import numpy as np
from itertools import combinations
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 115.0

    # Fallback: LUW Graph
    def get_luw(k, g):
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

    def check_circulant_girth(n, steps, g):
        # Non-backtracking walk check from vertex 0.
        # State: (node, prev) where prev is the node we came from.
        # Start: for each step s, go to s%n (prev=0) and (-s)%n (prev=0) if not 0.
        # If we ever reach node 0 with depth+1 < g, we have a cycle shorter than g.
        # Use deque for O(1) popleft.
        
        visited = set()
        queue = deque()
        for s in steps:
            v1 = s % n
            v2 = (-s) % n
            if v1 != 0:
                queue.append((v1, 0, 1))
                visited.add((v1, 0))
            if v2 != 0 and v2 != v1:
                queue.append((v2, 0, 1))
                visited.add((v2, 0))
        
        while queue:
            u, prev, depth = queue.popleft()
            if depth >= g - 1:
                continue
            for s in steps:
                v1 = (u + s) % n
                v2 = (u - s) % n
                for v in (v1, v2):
                    if v == prev:
                        continue
                    if v == 0:
                        if depth + 1 < g:
                            return False
                    elif (v, u) not in visited:
                        visited.add((v, u))
                        queue.append((v, u, depth + 1))
        return True

    if k == 4 and g == 10:
        # Search for circulant graphs
        # Moore bound: 1 + k*sum_{i=0}^{g/2-2} (k-1)^i = 1 + 4*(1+3+9+27) = 1+4*40 = 161
        # So n >= 162.
        
        # Strategy: Try random sampling for each n, with more samples for smaller n.
        # Also try structured candidates: steps that are "well-spread" in 1..n//2.
        
        # Pre-compute some heuristic step sets for each n
        def heuristic_steps(n):
            """Generate a few structured step sets likely to have high girth."""
            ms = n // 2
            candidates = []
            # Spread steps roughly evenly
            if ms >= 4:
                # Evenly spaced
                step_size = ms // 4
                base = [step_size, 2*step_size, 3*step_size, min(4*step_size, ms)]
                if len(set(base)) == 4:
                    candidates.append(sorted(set(base)))
                # Golden ratio-ish spacing
                phi = 1.618
                base2 = [1, 1 + int(phi*ms/4), 1 + int(2*phi*ms/4), 1 + int(3*phi*ms/4)]
                base2 = [min(x, ms) for x in base2]
                if len(set(base2)) == 4:
                    candidates.append(sorted(set(base2)))
            return candidates
        
        for n in range(162, 320):
            if time.time() > deadline:
                break
            
            max_step = n // 2
            if max_step < 4:
                continue
            
            # Try heuristic steps first
            for hsteps in heuristic_steps(n):
                if time.time() > deadline:
                    break
                if len(hsteps) == 4 and check_circulant_girth(n, hsteps, g):
                    edges = []
                    for u in range(n):
                        for s in hsteps:
                            v = (u + s) % n
                            if u < v:
                                edges.append((u, v))
                    return n, edges
            
            # Random search - more samples for smaller n
            # Budget: use remaining time / remaining n values
            remaining_time = deadline - time.time()
            remaining_ns = 320 - n
            if remaining_time <= 0:
                break
            # Adaptive sample count
            samples_per_n = max(50, min(5000, int(remaining_time * 100 / max(1, remaining_ns))))
            
            for _ in range(samples_per_n):
                if time.time() > deadline:
                    break
                steps = random.sample(range(1, max_step + 1), 4)
                if check_circulant_girth(n, steps, g):
                    edges = []
                    for u in range(n):
                        for s in steps:
                            v = (u + s) % n
                            if u < v:
                                edges.append((u, v))
                    return n, edges
                    
    # Fallback
    return get_luw(k, g)

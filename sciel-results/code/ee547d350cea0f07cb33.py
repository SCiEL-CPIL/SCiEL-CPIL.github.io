import time
import random
import sys
import numpy as np

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
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

    if k == 4 and g == 10:
        # Try to construct a voltage graph lift of the Petersen graph
        # with group Z_32 to get n=320.
        
        # Petersen graph vertices 0..9
        # Edges:
        # Outer pentagon: 0-1, 1-2, 2-3, 3-4, 4-0
        # Inner star: 5-7, 7-9, 9-6, 6-8, 8-5
        # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        
        # Map edge to index for voltage assignment
        edge_to_idx = {}
        for i, (u, v) in enumerate(pet_edges):
            edge_to_idx[(u, v)] = i
            edge_to_idx[(v, u)] = i
            
        # Precompute all closed walks of length 5 to 9 as coefficient vectors
        # coeff[j] is the net number of times edge j is traversed (with sign)
        # Actually, we just need the list of (idx, sign) for each walk
        closed_walks = []
        for L in range(5, 10):
            def gen_walk_dir(start, curr, length, path):
                if length == L:
                    if curr == start:
                        closed_walks.append(tuple(path))
                    return
                for idx, (u, v) in enumerate(pet_edges):
                    if curr == u:
                        gen_walk_dir(start, v, length + 1, path + [(idx, 1)])
                    elif curr == v:
                        gen_walk_dir(start, u, length + 1, path + [(idx, -1)])
                        
            for s in range(10):
                gen_walk_dir(s, s, 0, [])
                
        # Optimize: Convert walks to sparse representation for faster checking
        # A walk is valid if sum(sign * w[idx]) % 32 != 0
        # We can group by edge index?
        # For a specific walk, the sum is linear in w.
        
        # Backtracking solver
        n_edges = 15
        voltages = [0] * n_edges
        found = False
        deadline = time.time() + 100
        
        # Symmetry breaking: w[0] = 1
        # Note: if all voltages are 0, it's invalid.
        # If we fix w[0]=1, we cover the case where w[0]!=0.
        # What if w[0]=0 is required? Unlikely for girth.
        # We can try w[0] in 1..31 if needed, but 1 is a good start.
        
        def solve(idx, current_voltages):
            if time.time() > deadline:
                return False
            if idx == n_edges:
                # Check all constraints
                for walk in closed_walks:
                    total = 0
                    for e_idx, sign in walk:
                        total += sign * current_voltages[e_idx]
                    if total % 32 == 0:
                        return False
                voltages[:] = current_voltages
                return True
            
            for val in range(32):
                if idx == 0 and val == 0:
                    continue # Symmetry breaking
                current_voltages[idx] = val
                # Pruning: Check constraints that are fully determined by assigned vars
                # This is complex. Simple backtracking might be fast enough.
                # Number of walks is small.
                if solve(idx + 1, current_voltages):
                    return True
            return False

        if solve(0, voltages):
            found = True
            best_voltages = voltages[:]
        else:
            # Try without symmetry breaking if failed?
            # Or just fallback.
            found = False

        if found:
            # Construct the lift
            n = 10 * 32
            edges = []
            for u, v in pet_edges:
                idx = edge_to_idx[(u, v)]
                w = best_voltages[idx]
                for g_val in range(32):
                    u_idx = u * 32 + g_val
                    v_idx = v * 32 + (g_val + w) % 32
                    edges.append((u_idx, v_idx))
                    
            # Verify girth
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            def has_girth_at_least(n, edges, g):
                adj = [[] for _ in range(n)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                
                edge_set = set()
                for u, v in edges:
                    if u == v:
                        return False
                    if (v, u) in edge_set:
                        return False
                    edge_set.add((u, v))
                
                for start in range(n):
                    dist = [-1] * n
                    parent = [-1] * n
                    dist[start] = 0
                    queue = [start]
                    head = 0
                    while head < len(queue):
                        u = queue[head]
                        head += 1
                        if dist[u] >= g // 2:
                            continue
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                queue.append(v)
                            else:
                                if v != parent[u] and u != parent[v]:
                                    cycle_len = dist[u] + dist[v] + 1
                                    if cycle_len < g:
                                        return False
                return True
                
            if has_girth_at_least(n, edges, g):
                return n, edges

    # Fallback to LUW
    return luw_construct(k, g)

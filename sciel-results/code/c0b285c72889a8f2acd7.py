import time
import itertools
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Target: 4-regular, girth >= 10
    # Search for circulant graphs C_n(S)
    
    # Moore bound for (4,10) is 243.
    # n must be even.
    # Try n from 244 to 8000
    best_n = 8192 # Fallback to LUW size
    best_edges = None
    
    # We can also try to construct the LUW graph as a fallback
    # But let's try circulant first.
    
    def check_girth(n, steps):
        # steps is a set of 4 integers
        # Graph is on Z_n
        # Check for cycles of length < g
        # Use BFS from 0 to depth g//2 - 1?
        # Actually, just check all edges (u, v) for cycle length < g
        # in the BFS tree from 0.
        
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u in range(n):
            for s in steps:
                v1 = (u + s) % n
                v2 = (u - s) % n
                adj[u].append(v1)
                adj[u].append(v2)
        
        # Remove duplicates in adj lists to ensure simple graph
        # (Steps are distinct and not n/2, so v1 != v2 for each s, 
        #  and for different s1, s2, (u+s1)%n != (u+s2)%n)
        
        # BFS from 0
        parent = [-1] * n
        depth = [-1] * n
        parent[0] = 0
        depth[0] = 0
        queue = [0]
        head = 0
        
        min_cycle = float('inf')
        
        while head < len(queue):
            u = queue[head]
            head += 1
            if depth[u] >= g // 2: 
                # We only care about cycles < g
                # If depth is large, cycles formed here will be large
                # But we must check all edges
                pass
            
            for v in adj[u]:
                if depth[v] == -1:
                    depth[v] = depth[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Edge (u, v) exists in graph
                    # If u and v are in the BFS tree, 
                    # they form a cycle if they are not parent-child
                    # Cycle length = depth[u] + depth[v] + 1
                    # But we must ensure it's not the tree edge
                    if parent[u] == v or parent[v] == u:
                        continue
                    # Also, if u==v, it's a self loop (not possible)
                    # If u and v are siblings or cousins, cycle is formed
                    cycle_len = depth[u] + depth[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                    if min_cycle < g:
                        return min_cycle
        return min_cycle

    # Iterate n
    # To speed up, we can randomize steps or use heuristic
    # But we need to be thorough for small n
    
    found = False
    for n in range(244, 2000, 2):
        if time.time() > deadline:
            break
        
        # Generate combinations of 4 steps
        # Steps in 1..n//2
        # This is C(n/2, 4) which is large for n=1000 (25M)
        # We need a faster way or random sampling
        
        # Random sampling approach
        # Try many random sets of steps
        for _ in range(5000):
            if time.time() > deadline:
                break
            
            steps = random.sample(range(1, n//2 + 1), 4)
            # Ensure no step is n/2 (handled by range)
            # Ensure steps are distinct (sample handles)
            
            girth = check_girth(n, steps)
            if girth >= g:
                # Construct edges
                edges = []
                for u in range(n):
                    for s in steps:
                        v = (u + s) % n
                        if u < v:
                            edges.append((u, v))
                return n, edges
        
    # If no circulant found, return LUW (seed behavior)
    # Re-implement seed logic for fallback
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

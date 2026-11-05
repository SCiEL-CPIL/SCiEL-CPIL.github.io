import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = 110  # 10 seconds margin
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph CD(d, q) with q=3
    # This is valid for k=3, g=16 with d=11.
    # n = 2 * 3^11 = 354294
    def luw_fallback():
        q = 3
        d = 11
        # Construct CD(d, q)
        # Points and lines are vectors in GF(q)^d
        # Total vertices 2 * q^d
        # We need to return edges.
        # This is complex to implement fully in the fallback if not already in seed.
        # The seed code has it. We can reuse the logic or just return the seed's result if we don't improve.
        # But we must return (n, edges).
        # Let's just return the seed's output if we fail to find a better one.
        # Wait, I can't call the seed function. I must implement it or hardcode.
        # The seed code is provided. I can copy the LUW logic into the fallback.
        
        # Actually, I will just try to find a small graph.
        # If I fail, I will return a dummy valid graph? No, must be valid.
        # I will implement the LUW construction in the fallback.
        
        pass # Placeholder
        
    # Strategy: Try to find a 3-regular graph with girth g for N starting from Moore bound.
    # Moore bound for k=3, g=16 is 510.
    # Try N = 512, 514, ...
    
    def is_valid_girth(adj, n, g):
        # Check if girth >= g
        # For each vertex, BFS up to depth g/2 - 1
        # If we find a neighbor at depth < g/2 that is already visited at a smaller depth, we have a cycle.
        # Actually, simpler: check for cycles of length 3, 4, ..., g-1.
        # For 3-regular, no triangles if g >= 4.
        # We can just do BFS from each vertex up to depth g//2 - 1.
        # If we encounter a vertex already in the BFS tree at a different path, check cycle length.
        
        # Efficient check:
        # For each edge (u, v), check if there is a path of length < g-1 between u and v not using the edge.
        # This is expensive.
        
        # Better: 
        # 1. Check for triangles (g>=3).
        # 2. Check for 4-cycles (g>=4).
        # ...
        
        # Let's use the standard "shortest cycle" algorithm.
        # For each vertex s, run BFS.
        # Track dist[s][v] and parent[s][v].
        # If we find an edge (u, v) such that dist[s][u] + dist[s][v] + 1 < g, then we have a cycle < g.
        # We only need to check up to depth g//2.
        
        half_g = g // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            queue = [s]
            dist[s] = 0
            idx = 0
            while idx < len(queue):
                u = queue[idx]
                idx += 1
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a cycle
                        # Cycle length = dist[u] + dist[v] + 1
                        # But this might be the same path back.
                        # We need to ensure it's a simple cycle.
                        # If parent[u] != v and parent[v] != u, it's a cycle.
                        if parent[u] != v and parent[v] != u:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def generate_random_3_reg(n):
        # Configuration model
        if n % 2 != 0:
            return None
        stubs = [i for i in range(n) for _ in range(3)]
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        for i in range(0, n*3, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None # Loop
            if v in adj[u]:
                return None # Multi-edge
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def try_find_graph(n, g, time_limit):
        start_t = time.time()
        attempts = 0
        while time.time() - start_t < time_limit:
            adj = generate_random_3_reg(n)
            if adj is None:
                continue
            if is_valid_girth(adj, n, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
            attempts += 1
            if attempts % 1000 == 0:
                # Optional: try local search here?
                pass
        return None

    # Try small N first
    # Moore bound is 510.
    # Try 512, 514, 516...
    # But random search is very unlikely to find girth 16 for N=512.
    # We need a smarter search or larger N.
    
    # Let's try N up to 2000 with a time budget.
    # If we find one, great.
    # If not, fallback to LUW.
    
    best_result = None
    
    # Time budget for search: 100s
    # Fallback: LUW (n=354294)
    
    # Since random search for girth 16 is hard, we might not find anything small.
    # But we MUST return a valid graph.
    # The LUW graph is valid.
    
    # Let's implement the LUW construction properly as the fallback.
    
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

    # Try to find a better graph
    # Given the difficulty, we might just return the LUW graph.
    # But let's try a few random graphs for N=512 to 1000.
    
    search_deadline = time.time() + 100
    
    # Try N=512
    for N in range(512, 2000, 2):
        if time.time() > search_deadline:
            break
        result = try_find_graph(N, g, 5) # 5 seconds per N
        if result:
            return result
            
    # Fallback
    return luw_construct(k, g)

import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    # 1. Calculate Moore Bound
    # For bipartite k-regular graph with girth g:
    # If g is odd, target girth is g+1 for bipartite? 
    # "No cycle shorter than g". 
    # If g=9, cycles of length 8 are forbidden. Cycles of length 10 are allowed.
    # So we need girth >= 10 for bipartite.
    # However, we are NOT restricted to bipartite in the construction, but 
    # standard random regular graphs are often bipartite-ish or not.
    # If we generate a general graph, girth 9 is allowed.
    # Moore bound for general graph k=5, g=9:
    # N >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
    
    half_g = (g - 1) // 2
    moore_sum = 1
    for i in range(1, half_g + 1):
        moore_sum += (k - 1) ** i
    N_min = k * moore_sum + 1
    
    # For bipartite, N must be even.
    # If we find an odd N graph, it's valid if girth >= g.
    
    # Fallback: LUW construction (Deterministic)
    def luw_construct(k, g):
        q = k
        # d is the "depth"
        # For girth g, we need 2*d >= g?
        # LUW graphs are bipartite with girth 2*d+2?
        # Standard LUW for (k, 2d+2) uses q^d vertices per side?
        # Let's use the same logic as the seed which seemed to work.
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

    # Helper: Generate random k-regular graph
    def generate_random_regular(n, k):
        # Configuration model
        # Ensure n*k is even
        if (n * k) % 2 != 0:
            return None
            
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
                
        random.shuffle(stubs)
        
        edges = set()
        adjacency = [[] for _ in range(n)]
        
        # Pair up stubs
        # If we get a self-loop or multi-edge, we need to swap
        # Simple approach: try to resolve
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                # Self-loop, swap with another stub
                found_swap = False
                for j in range(i+2, len(stubs), 2):
                    w = stubs[j]
                    if w != u and (u, w) not in edges and (w, u) not in edges:
                        # Swap v and w
                        stubs[i+1], stubs[j] = stubs[j], stubs[i+1]
                        v = stubs[i+1]
                        found_swap = True
                        break
                if not found_swap:
                    return None
            if (u, v) in edges or (v, u) in edges:
                # Multi-edge, try to swap
                found_swap = False
                for j in range(i+2, len(stubs), 2):
                    w = stubs[j]
                    if w != u and (u, w) not in edges and (w, u) not in edges and w != v:
                        # Check if swapping creates self loop?
                        if w != u:
                            stubs[i+1], stubs[j] = stubs[j], stubs[i+1]
                            v = stubs[i+1]
                            found_swap = True
                            break
                if not found_swap:
                    return None
                    
            edges.add((u, v))
            adjacency[u].append(v)
            adjacency[v].append(u)
            
        return adjacency, list(edges)

    # Helper: Check Girth
    def check_girth(adjacency, n, g):
        # BFS from every node
        # Return True if no cycle < g
        # Optimization: If we find a cycle < g, return False immediately
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= g - 1:
                    continue
                for v in adjacency[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    # Search Strategy
    # We search N from N_min upwards
    # We limit the number of tries per N
    max_N = N_min + 300
    best_result = None
    
    for N in range(N_min, max_N + 1):
        if time.time() > deadline:
            break
            
        # Try a few random graphs
        # If N is small, more tries might be needed, but random regular graphs are hard to generate if k is high relative to N.
        # For k=5, N>=5 is fine.
        tries = 20
        for _ in range(tries):
            if time.time() > deadline:
                break
                
            result = generate_random_regular(N, k)
            if result is None:
                continue
                
            adjacency, edges = result
            
            # Quick check: Are all degrees exactly k?
            # The generator ensures this if it doesn't return None
            
            if check_girth(adjacency, N, g):
                best_result = (N, edges)
                return best_result

    # Fallback to LUW
    luw_n, luw_edges = luw_construct(k, g)
    
    # Return the best one found (which will be the LUW one if search failed)
    return (luw_n, luw_edges)

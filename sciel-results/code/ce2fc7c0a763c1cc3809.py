import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Moore bound calculation for bipartite (if we enforce bipartite)
    # For bipartite girth g (even), Moore bound is 2 * sum_{i=0}^{g/2-1} (k-1)^i
    # Here g=9 is odd. If we enforce bipartite, we can only get even girths.
    # So we must allow odd cycles?
    # If we allow odd cycles, BFS is more complex.
    # However, most small cages for odd girth are not bipartite.
    # But checking for odd cycles of length < 9 (3,5,7) is required.
    
    # Let's try a non-bipartite greedy.
    # To speed up, we will use a simple adjacency list and BFS.
    
    best_n = float('inf')
    best_edges = []
    
    # Heuristic for initial N
    # Moore bound for k=5, g=9: 1 + 5*(1 + 5 + 25 + 125) = 781
    # Let's try N from 800 to 1200
    
    for trial in range(50):
        if time.time() > deadline:
            break
            
        # Try different N values
        # Start with N close to Moore bound, increase if stuck
        # We'll try to construct a graph with N vertices.
        # We need k*N to be even.
        
        # We will try a few N values
        for N in [800, 900, 1000, 1100, 1200]:
            if time.time() > deadline:
                break
            
            # Initialize
            adj = [set() for _ in range(N)]
            degrees = [0] * N
            edges = []
            
            # We need to add k*N/2 edges
            target_edges = k * N // 2
            
            # Pre-calculate target degree for each vertex
            # We will pick vertices with lowest degree
            
            success = True
            for step in range(target_edges):
                if time.time() > deadline:
                    success = False
                    break
                
                # Find vertex with min degree
                # Optimization: Use a heap? Or just scan. N=1000, scan is 1000 ops.
                # 2500 steps * 1000 = 2.5M ops. Fast.
                min_deg = k
                min_idx = -1
                for i in range(N):
                    if degrees[i] < min_deg:
                        min_deg = degrees[i]
                        min_idx = i
                
                if min_idx == -1:
                    success = False
                    break
                
                # Find candidate neighbors for min_idx
                # We want to connect min_idx to a vertex j such that:
                # 1. j != min_idx
                # 2. j not already connected to min_idx
                # 3. degrees[j] < k
                # 4. No cycle of length < g is formed.
                
                # Get candidates
                candidates = [j for j in range(N) if j != min_idx and degrees[j] < k and j not in adj[min_idx]]
                
                if not candidates:
                    success = False
                    break
                
                # Shuffle to break symmetry
                random.shuffle(candidates)
                
                valid_found = False
                for j in candidates:
                    # Check if adding (min_idx, j) creates a short cycle.
                    # This is equivalent to checking if there is a path between min_idx and j
                    # of length < g-1 in the current graph.
                    # Since we are adding the edge, the cycle length is path_len + 1.
                    # We need path_len + 1 >= g => path_len >= g-1.
                    # So we check if shortest path < g-1.
                    
                    # BFS from min_idx to j, up to depth g-2
                    max_depth = g - 2
                    visited = [-1] * N
                    visited[min_idx] = 0
                    queue = [(min_idx, 0)]
                    found_short = False
                    
                    while queue:
                        u, dist = queue.pop(0)
                        if dist >= max_depth:
                            continue
                        for v in adj[u]:
                            if visited[v] == -1:
                                visited[v] = dist + 1
                                if v == j:
                                    found_short = True
                                    break
                                queue.append((v, dist + 1))
                            if found_short:
                                break
                        if found_short:
                            break
                    
                    if not found_short:
                        # Add edge
                        adj[min_idx].add(j)
                        adj[j].add(min_idx)
                        degrees[min_idx] += 1
                        degrees[j] += 1
                        edges.append((min_idx, j))
                        valid_found = True
                        break
                
                if not valid_found:
                    success = False
                    break
            
            if success:
                # Verify degrees
                if all(d == k for d in degrees):
                    if N < best_n:
                        best_n = N
                        best_edges = edges[:]
                        # If we found a very good one, we can break early?
                        # Or keep searching for even smaller?
                        # Let's continue searching for smaller N in next trials?
                        # We are iterating N in increasing order.
                        # So if we succeed with N=800, it's likely the best for this trial.
                        # But maybe another trial with N=800 fails, and N=810 succeeds?
                        # We should try to find the minimum N.
                        # Break inner loop?
                        # If we found N=800, we can't do better than 800 (Moore bound 781).
                        # So we can break.
                        break
            
            # If we failed with N=800, try N=900, etc.
            # But if we succeeded with N=1000, we might find N=900 in next trial.
            # So we don't break the outer trial loop immediately unless best_n is very close to Moore bound.
        
        # If we found a graph with N <= 850, it's very good.
        if best_n <= 850:
            break

    if best_n == float('inf'):
        # Fallback: Return a valid graph even if large
        # Use the LUW construction from the seed program as fallback
        # But we need to implement it or return a simple valid graph.
        # Let's just return a cycle or something? No, must be k-regular.
        # We can generate a random regular graph and hope?
        # Or just return the LUW result.
        # Since I don't have the LUW code here, I'll implement a simple fallback:
        # A disjoint union of K_{k+1}? No, girth 4.
        # Let's just return a valid graph constructed by a simple method if greedy fails.
        # For now, if greedy fails, we return a large valid graph.
        # We can use the LUW logic if we copy it.
        # Let's copy the LUW logic from the prompt to ensure validity.
        
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
        index, order, edges_fallback = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges_fallback.append((index[vertex], index[other]))
        return len(order), edges_fallback

    return best_n, best_edges

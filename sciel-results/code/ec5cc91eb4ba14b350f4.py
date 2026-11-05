import time
import sys
from collections import deque
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin

    # --- Girth check using BFS from every vertex ---
    def check_girth(adj, n, g):
        half_g = (g - 1) // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if dist[u] <= half_g and dist[v] <= half_g:
                            if parent[u] == v or parent[v] == u:
                                continue
                            return False
        return True

    # --- LUW construction (baseline) ---
    def construct_luw(k, g):
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

    baseline_n, baseline_edges = construct_luw(k, g)
    best_n = baseline_n
    best_edges = baseline_edges

    # --- New strategy: Random 3-regular graph generation + girth check ---
    # This is a completely different approach: instead of using algebraic constructions
    # or voltage lifts, we generate random 3-regular graphs and check their girth.
    # The key insight: for small n, random 3-regular graphs often have high girth.
    # We can try various n values and keep the smallest valid one.
    
    def generate_random_3_regular(n):
        """Generate a random 3-regular graph on n vertices using configuration model."""
        if n < 4 or n % 2 != 0:
            return None
        
        # Create stubs: each vertex has 3 stubs
        stubs = []
        for i in range(n):
            for _ in range(3):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        # Pair up stubs to form edges
        edges = set()
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None  # Self-loop, retry
            if (min(u,v), max(u,v)) in edges:
                return None  # Multi-edge, retry
            edges.add((min(u,v), max(u,v)))
        
        if len(edges) != n * 3 // 2:
            return None
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Verify 3-regular
        for i in range(n):
            if len(adj[i]) != 3:
                return None
        
        return adj, list(edges)
    
    # Try random 3-regular graphs for various n
    # Start from small n and increase
    # The Moore bound for k=3, g=16 is roughly 2*(3^7 - 1)/2 + 1 = 3^7 = 2187? 
    # Wait, Moore bound for (3,16): n >= 2*(1 + 2 + 4 + ... + 2^7) = 2*(2^8 - 1) = 510? 
    # Actually for odd girth, it's different. For even girth g=16:
    # n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i = 2 * sum_{i=0}^{7} 2^i = 2*(2^8 - 1) = 510
    # So n must be at least 510. But the best known is 936.
    # Let's try n from 510 up to baseline_n.
    
    if k == 3:
        min_n = 510  # Moore bound approximation
        max_n = baseline_n
        
        # We'll try n values that are even (required for 3-regular)
        # Start from min_n, go up
        # For each n, try a few random graphs
        
        attempts_per_n = 20
        search_deadline = deadline - 3.0
        
        for n in range(min_n, max_n + 1, 2):
            if time.time() > search_deadline:
                break
            
            for _ in range(attempts_per_n):
                if time.time() > search_deadline:
                    break
                
                result = generate_random_3_regular(n)
                if result is None:
                    continue
                
                adj, edges = result
                
                if check_girth(adj, n, g):
                    if n < best_n:
                        best_n = n
                        best_edges = edges
                    break  # Found valid graph for this n, try smaller n? No, n is increasing.
                           # Actually we want the smallest n, so if we find one at n, 
                           # we should stop because smaller n was already tried.
                           # Wait, we're going from min_n up. So the first success is the best.
                           # But we might have missed a valid graph at a smaller n due to bad luck.
                           # So we should continue to see if we can find a smaller one? No, 
                           # we're already at n, and smaller n's were tried.
                           # So we can break out of the n loop? No, because we might find 
                           # a valid graph at a larger n but a smaller n might also work 
                           # if we try more attempts. But we already tried attempts_per_n for each smaller n.
                           # So let's just break and return the best found.
                           # Actually, since n is increasing, the first success gives the smallest n.
                           # So we can break out of both loops.
                    break
            
            if best_n < baseline_n:
                # Found an improvement. Since n is increasing, we can stop.
                break
    
    return best_n, best_edges

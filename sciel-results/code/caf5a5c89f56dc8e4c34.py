import time
import math
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s buffer for safety

    def luw_construction(d, q):
        # ... (Keep existing LUW code) ...
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
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

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

    def get_girth(n, edges, target_girth):
        if n == 0:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # BFS from each vertex to find shortest cycle
        # Optimization: Only need to check up to depth target_girth//2
        max_depth = target_girth // 2
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] > max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] <= max_depth:
                            queue.append(v)
                    else:
                        # If v is not the parent of u, we found a cycle
                        # The length is dist[u] + dist[v] + 1
                        # But this logic is slightly flawed for undirected graphs in BFS
                        # Standard BFS girth check:
                        # When visiting v from u:
                        # If dist[v] == -1, set dist[v] = dist[u]+1
                        # Else, if v != parent[u], cycle length = dist[u] + dist[v] + 1
                        # We need to ensure we don't count the edge (u,v) as a self-loop if v==u (no loops in simple graph)
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < target_girth:
                                return False
        return True

    def random_search(k, g, min_n, max_n, time_limit):
        """
        Searches for a k-regular graph with girth >= g 
        with n vertices in [min_n, max_n].
        """
        best_result = None
        best_n = float('inf')
        
        # Start from min_n and go up
        # Step size 2 to ensure parity if needed? 
        # For k-regular, n*k must be even.
        # If k is odd, n must be even.
        
        current_n = min_n
        if k % 2 == 1 and current_n % 2 != 0:
            current_n += 1
            
        while current_n <= max_n and time.time() < time_limit:
            # Try a few random graphs for this n
            attempts = 0
            found = False
            while attempts < 50 and time.time() < time_limit:
                attempts += 1
                # Generate random k-regular graph
                # Configuration Model
                stubs = []
                for i in range(current_n):
                    for _ in range(k):
                        stubs.append(i)
                random.shuffle(stubs)
                
                edges = []
                valid = True
                edge_set = set()
                
                for i in range(0, len(stubs), 2):
                    u = stubs[i]
                    v = stubs[i+1]
                    if u == v:
                        valid = False
                        break
                    # Normalize edge
                    if u > v:
                        u, v = v, u
                    if (u, v) in edge_set:
                        valid = False
                        break
                    edge_set.add((u, v))
                    edges.append((u, v))
                    
                if not valid:
                    continue
                    
                # Check girth
                if get_girth(current_n, edges, g):
                    if current_n < best_n:
                        best_n = current_n
                        best_result = (current_n, edges)
                    found = True
                    break # Move to next n? Or try to find smaller? 
                    # Since we are increasing n, the first valid one is the smallest for this n.
                    # But we want global min. So we should return immediately?
                    # No, we started from min_n. So the first valid n is the global min found so far.
                    # We can return immediately?
                    # Yes, because we are searching n increasing.
                    return best_result
            
            current_n += 2 # Ensure even if k is odd? 
            # Actually, if k is even, n can be anything.
            # If k is odd, n must be even.
            if k % 2 == 1:
                current_n += 0 # already even
            else:
                current_n += 1 # try next integer

        return best_result

    # Determine candidate d values for LUW
    q = k
    min_d = max(1, (g - 4 + 1) // 2)
    seed_d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    candidates = sorted(set([min_d, min_d + 1, min_d + 2, seed_d]))
    
    best_result = None
    best_n = float('inf')
    
    # 1. Try LUW constructions
    for d in candidates:
        if time.time() > deadline - 20:
            break
        try:
            n, edges = luw_construction(d, q)
        except Exception:
            continue
            
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(dg != k for dg in deg):
            continue
            
        # Check girth
        if not get_girth(n, edges, g):
            continue
            
        if n < best_n:
            best_n = n
            best_result = (n, edges)
            
    # 2. If LUW didn't find a very good graph, try Random Search
    # Moore Bound
    if k > 1:
        moore_bound = 1 + k * sum((k-1)**i for i in range(g//2))
    else:
        moore_bound = 2 * (g//2) + 1
        
    # Only start random search if LUW result is not already very close to Moore bound
    # Or if LUW failed.
    # Let's try random search for n in [moore_bound, min(best_n, 1000)]
    # If best_n is inf, go up to 1000.
    
    upper_bound_random = min(best_n if best_n < float('inf') else 1000, 1000)
    
    if moore_bound < upper_bound_random and time.time() < deadline - 10:
        # Search for better graph using random method
        # We want to find n < best_n
        # So we search from moore_bound to best_n - 1
        rand_result = random_search(k, g, moore_bound, best_n - 1, deadline - 5)
        if rand_result:
            if rand_result[0] < best_n:
                best_result = rand_result
                best_n = rand_result[0]

    if best_result:
        return best_result
    else:
        # Fallback to seed's logic if all else fails
        d = seed_d
        return luw_construction(d, q)

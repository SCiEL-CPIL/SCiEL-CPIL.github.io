import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 140.0 # Leave 10s buffer for 150s limit
    
    # Moore bound calculation
    # For k=3, g=16: 1 + 3 + 3*2 + 3*2^2 + 3*2^3 + 3*2^4 + 3*2^5 + 3*2^6 (if g even, depth g/2)
    # Girth g means no cycle < g.
    # Tree of depth g/2 - 1 has no cycles.
    # Depth D = g // 2
    # If g is even, max depth without cycle is g/2 - 1?
    # Cycle of length 2D implies depth D.
    # So we can have depth D-1.
    # Moore bound N_min = 1 + k * sum_{i=0}^{D-2} (k-1)^i for even g?
    # Actually, standard Moore bound:
    # If g is even, D = g/2. N >= 1 + k * sum_{i=0}^{D-2} (k-1)^i
    # If g is odd, D = (g-1)/2. N >= 1 + k * sum_{i=0}^{D-1} (k-1)^i
    
    if g % 2 == 0:
        D = g // 2
        # Sum for i=0 to D-2
        terms = D - 1
        if terms <= 0:
            n_min = 1 + k
        else:
            n_min = 1 + k * sum((k-1)**i for i in range(terms))
    else:
        D = (g - 1) // 2
        terms = D
        n_min = 1 + k * sum((k-1)**i for i in range(terms))
        
    n_min = max(n_min, k + 1) # Safety
    
    best_graph = None
    best_n = float('inf')
    
    # Search range
    # Start from n_min, go up to n_min + 1000 or until time runs out
    # We want to find the SMALLEST n with girth >= g
    # So we should try small n first.
    
    # However, finding a graph with girth g for small n is hard.
    # Maybe start with a larger n where it's easier, then try to shrink?
    # No, the score is 1/n. Smaller n is better.
    # So we must find a valid graph.
    
    # Strategy:
    # 1. Try n from n_min to n_min + 500.
    # 2. For each n, try to find a valid graph.
    # 3. If found, return it.
    
    # To increase success rate, we can run multiple random restarts for a given n.
    
    def generate_random_regular(n, k):
        # Configuration model
        if n * k % 2 != 0:
            return None
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        adj = {i: set() for i in range(n)}
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                # Self loop, try to fix or restart
                # Simple approach: restart if self loop
                return None
            if v in adj[u]:
                # Multi-edge, restart
                return None
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def check_girth(adj, n, target_g):
        # BFS from each node
        min_girth = float('inf')
        for start in range(n):
            visited = {start: 0}
            parent = {start: -1}
            q = deque([start])
            while q:
                u = q.popleft()
                if visited[u] > target_g // 2:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if v in visited:
                        # Found a cycle
                        # Length = visited[u] + 1 + visited[v] ? No.
                        # Path start->u and start->v.
                        # Cycle length = dist(start, u) + dist(start, v) + 1
                        # But this only works if LCA is start?
                        # No, standard BFS girth check:
                        # When we see a visited node v (not parent), 
                        # cycle length = dist[u] + 1 + dist[v] is NOT correct if LCA is not start.
                        # Correct way: 
                        # When exploring edge (u,v), if v is visited and v != parent[u]:
                        # If v is in current BFS level or previous, we found a cycle.
                        # Actually, simpler: 
                        # BFS tree. If we encounter a visited node, the cycle length is 
                        # dist[u] + dist[v] + 1 ONLY IF the paths don't share nodes other than start.
                        # This is tricky.
                        
                        # Robust method: 
                        # For each start, BFS. 
                        # If we find a back edge (u,v), the cycle length is not trivially dist[u]+dist[v]+1.
                        # We need to find the LCA of u and v in the BFS tree.
                        # This is O(N) per edge.
                        
                        # Alternative: 
                        # Just compute all-pairs shortest paths? O(N^3).
                        # Or, for each start, do BFS and record the first time a node is visited.
                        # If we visit a node already visited, we found a cycle.
                        # The length of the cycle formed by the tree path from start to u, 
                        # the edge (u,v), and the tree path from v to start?
                        # No, that's a cycle of length dist[u] + 1 + dist[v].
                        # This is a valid cycle. It might not be the shortest one involving u,v, 
                        # but it is a cycle.
                        # We want the MINIMUM cycle length.
                        # Taking min over all back edges found in BFS from all starts 
                        # gives the girth.
                        
                        cycle_len = visited[u] + 1 + visited[v]
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            if min_girth < target_g:
                                return min_girth # Early exit
                    else:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        q.append(v)
        return min_girth

    def find_shortest_cycle(adj, n):
        # Returns the cycle as a list of nodes
        # Use BFS from each node, track parents
        for start in range(n):
            visited = {start: 0}
            parent = {start: -1}
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if v in visited:
                        # Reconstruct cycle
                        cycle = [u, v]
                        # Walk back from u to LCA
                        u_curr = u
                        v_curr = v
                        # Find LCA
                        # Since BFS, we can just walk back
                        # But we need the actual path
                        path_u = [u]
                        while path_u[-1] != v:
                            u_curr = parent[u_curr]
                            if u_curr == -1: break
                            path_u.append(u_curr)
                        # This logic is flawed if LCA is not start.
                        # Simplified: Just return the length for now.
                        # For the heuristic, we just need to break a short cycle.
                        # We can just pick a random edge from a short cycle.
                        pass
                    else:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        q.append(v)
        return None

    # Search Loop
    max_n_to_try = n_min + 1000
    for n in range(n_min, max_n_to_try):
        if time.time() > deadline:
            break
            
        # Try to find a valid graph of size n
        # Run multiple restarts
        attempts = 0
        max_attempts = 100
        found = False
        
        while attempts < max_attempts and time.time() < deadline:
            adj = generate_random_regular(n, k)
            if adj is None:
                attempts += 1
                continue
            
            girth = check_girth(adj, n, g)
            if girth >= g:
                # Found a valid graph
                # Convert to edge list
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
            
            attempts += 1
            
    # If no graph found in the search range, fall back to LUW
    # Re-implement LUW concisely
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

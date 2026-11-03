import time
import sys
import collections
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # We focus on k=3
    if k != 3:
        # Fallback for other k: LUW
        return _construct_luw(k, g, deadline)

    # Moore bound for k=3, g=16 is 382.
    # Best known is 936.
    # We search for Generalized Petersen Graphs GP(n, k_gp) with 2n vertices.
    # We want 2n to be small.
    
    def check_girth(n, edges, min_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        max_depth = min_girth // 2 - 1
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = collections.deque([start])
            
            while queue:
                curr = queue.popleft()
                if dist[curr] > max_depth:
                    continue
                for neighbor in adj[curr]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[curr] + 1
                        parent[neighbor] = curr
                        queue.append(neighbor)
                    elif neighbor != parent[curr]:
                        # Found a cycle
                        if dist[curr] + dist[neighbor] + 1 < min_girth:
                            return False
        return True

    def gp_girth_filter(n, k_gp, g):
        """
        Quick check for short cycles in GP(n, k_gp).
        Returns False if a short cycle is detected.
        """
        half = g // 2
        
        # 1. Check inner cycle: k_gp^j mod n
        # If k_gp^j == 1 or -1 for j < half, cycle exists.
        val = 1
        for j in range(1, half):
            val = (val * k_gp) % n
            if val == 1 or val == n - 1:
                return False
        
        # 2. Check spokes: i + k_gp^j == -i mod n => 2i + k_gp^j == 0 mod n
        # This requires solving for i.
        # If gcd(2, n) does not divide k_gp^j, no solution.
        # If gcd(2, n) divides k_gp^j, there is a solution.
        # The cycle length would be 2j + 2.
        # We need 2j + 2 >= g => j >= g/2 - 1.
        # So if there is a solution for j < g/2 - 1, it's bad.
        
        g2 = math.gcd(2, n)
        for j in range(1, half - 1):
            val = pow(k_gp, j, n)
            if val % g2 == 0:
                return False
                
        # 3. Check outer-inner paths: i + 1 + k_gp^j == i mod n => 1 + k_gp^j == 0 mod n
        # Cycle length 2j + 2.
        # Need j < g/2 - 1.
        val = 1
        for j in range(1, half - 1):
            val = (val * k_gp) % n
            if val == n - 1:
                return False
                
        return True

    best_n = float('inf')
    best_edges = None

    # Search GP graphs
    # n is the number of vertices in the outer cycle. Total vertices = 2n.
    # We want 2n to be small.
    # Moore bound 382 => 2n >= 382 => n >= 191.
    # Known best 936 => n = 468.
    # We search n from 191 to 600.
    
    # To find the smallest n, we iterate n in increasing order.
    # However, not every n has a valid k_gp.
    # We can iterate n and check k_gp.
    
    max_n_search = 600
    for n in range(191, max_n_search + 1):
        if time.time() > deadline - 20.0:
            break
        
        for k_gp in range(1, n // 2 + 1):
            if time.time() > deadline - 15.0:
                break
            
            if math.gcd(n, k_gp) != 1:
                continue
            
            # Apply filter
            if not gp_girth_filter(n, k_gp, g):
                continue
            
            edges_gp = []
            # Outer cycle
            for i in range(n):
                edges_gp.append((i, (i + 1) % n))
            # Spokes
            for i in range(n):
                edges_gp.append((i, n + i))
            # Inner cycle
            for i in range(n):
                edges_gp.append((n + i, n + (i + k_gp) % n))
            
            if check_girth(2 * n, edges_gp, g):
                return 2 * n, edges_gp

    # Fallback: LUW
    # q=3, d=7 (girth >= 16)
    return _construct_luw(3, g, deadline)

def _construct_luw(k, g, deadline):
    # LUW construction
    # For k=3, q=3.
    # Girth >= 2d+2.
    # Need 2d+2 >= g => d >= (g-2)/2.
    d = max(3, (g - 2) // 2 + 1)
    
    q = k
    if q % 2 == 1 and d % 2 == 0:
        d += 1
        
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]

    index = {}
    order = []
    edges = []
    
    start = (0, (None,) + (0,) * d)
    index[start] = 0
    order.append(start)
    
    for i, vertex in enumerate(order):
        if time.time() > deadline - 1.0:
            break
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            
            neighbor = (1 - side, tuple(y))
            
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            
            if side == 0:
                edges.append((index[vertex], index[neighbor]))
                
    n_luw = len(order)
    return n_luw, edges

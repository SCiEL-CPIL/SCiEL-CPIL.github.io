"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Strategy 1: Try to find a Cayley graph on an abelian group with n close to Moore bound or known best.
    # For k=3, g=16, known best is 936.
    # Try groups of order N where N is a multiple of k? No, N must be such that a 3-regular Cayley graph exists.
    # A 3-regular Cayley graph on an abelian group requires a connection set S of size 3.
    # If the group has an element of order 2, S can be {a, b, -b} with a=-a.
    # Or S = {a, b, c} with a+b+c=0? No, Cayley graphs are defined by S.
    
    def check_girth(adj, n, girth_target):
        # BFS from each node, check for cycles < girth_target
        # Optimization: only need to check up to depth girth_target//2 - 1
        limit = girth_target // 2 - 1
        for i in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[i] = 0
            queue = [i]
            for u in queue:
                if dist[u] >= limit:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < girth_target:
                            return False
        return True

    def build_cayley_graph(group_order, gens):
        # group is Z_m x Z_n or similar
        # This is a simplified version for Z_m x Z_n
        pass

    # Specific attempt for k=3, g=16
    if k == 3 and g == 16:
        # Try to construct the known 936-vertex cage or a smaller one.
        # Since 936 is the known best, and Moore bound is 512, we aim for <= 936.
        
        # Let's try a specific construction: The (3,16)-cage is a Cayley graph on Z_13 x Z_72?
        # Or Z_3 x Z_312?
        
        # We will try a few abelian groups.
        candidates = [
            (13, 72),
            (3, 312),
            (12, 78),
            (26, 36),
            (78, 12),
            (312, 3),
            (936, 1) # Z_936
        ]
        
        best_n = float('inf')
        best_edges = None
        
        for m, n in candidates:
            if m * n > 1000: # Keep search space small
                continue
                
            # Try to find generators
            # For Z_m x Z_n, element is (a, b).
            # Generators g1, g2, g3.
            # Degree 3: S = {g1, g2, g3} such that g1=-g1? No.
            # If we use S = {g1, g2, -g2}, we need g1 to be an involution?
            # If g1 is not an involution, degree is 6? No.
            # Cayley graph degree is |S|.
            # If S = {a, b, c}, degree is 3.
            # For the graph to be simple and undirected, we need S = -S?
            # No, if S is not symmetric, the graph is directed.
            # The task requires a simple undirected graph.
            # So S must be symmetric: if x in S, -x in S.
            # If |S|=3, this is impossible in an abelian group unless one element is its own inverse (order 2) and the other two are inverses.
            # So S = {u, v, -v} where u = -u (order 2).
            
            # Find an element u of order 2 in Z_m x Z_n.
            # u = (m/2, 0) if m is even, or (0, n/2) if n is even, or (m/2, n/2).
            
            u = None
            if m % 2 == 0:
                u = (m // 2, 0)
            elif n % 2 == 0:
                u = (0, n // 2)
            else:
                continue # No element of order 2? (Both odd? 13*72 is even. 3*312 is even.)
            
            # Now find v such that girth is large.
            # v = (a, b).
            # We need to search for v.
            
            # To limit search, try a few random v's.
            found = False
            for _ in range(1000):
                if time.time() > deadline:
                    break
                a = random.randint(1, m - 1)
                b = random.randint(1, n - 1)
                v = (a, b)
                
                # Check if v is independent from u
                # Build graph
                # Vertex (x, y)
                # Neighbors: (x+u_x, y+u_y), (x+v_x, y+v_y), (x-v_x, y-v_y) mod (m, n)
                
                adj = [[] for _ in range(m * n)]
                edges = []
                
                for x in range(m):
                    for y in range(n):
                        idx = x * n + y
                        # Neighbor 1: +u
                        nx = (x + u[0]) % m
                        ny = (y + u[1]) % n
                        nidx = nx * n + ny
                        if nidx not in adj[idx]:
                            adj[idx].append(nidx)
                            adj[nidx].append(idx)
                            edges.append((idx, nidx))
                            
                        # Neighbor 2: +v
                        nx = (x + v[0]) % m
                        ny = (y + v[1]) % n
                        nidx = nx * n + ny
                        if nidx not in adj[idx]:
                            adj[idx].append(nidx)
                            adj[nidx].append(idx)
                            edges.append((idx, nidx))
                            
                        # Neighbor 3: -v
                        nx = (x - v[0]) % m
                        ny = (y - v[1]) % n
                        nidx = nx * n + ny
                        if nidx not in adj[idx]:
                            adj[idx].append(nidx)
                            adj[nidx].append(idx)
                            edges.append((idx, nidx))
                
                # Check degrees
                valid_deg = True
                for i in range(m * n):
                    if len(adj[i]) != 3:
                        valid_deg = False
                        break
                if not valid_deg:
                    continue
                    
                # Check girth
                if check_girth(adj, m * n, g):
                    return m * n, edges
            
        # If no Cayley graph found, fallback to LUW
        pass

    # Fallback: LUW Construction (Fixed)
    q = k
    d = max(3, (g - 2) // 2)
    
    while True:
        if time.time() > deadline:
            # Return best effort or raise
            break
            
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
                    
        n = len(order)
        
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        valid = True
        limit_depth = g // 2 - 1
        
        for i in range(n):
            if time.time() > deadline:
                break
            dist = [-1] * n
            parent = [-1] * n
            dist[i] = 0
            queue = [i]
            for u in queue:
                if dist[u] >= limit_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            valid = False
                            break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return n, edges
            
        d += 1
        if d > 15:
             break

    # Emergency fallback: Return a trivial valid graph if possible?
    # Or raise?
    # The task requires a valid graph.
    # If we failed, we should try to return the last valid one or a known small one.
    # For k=3, g=16, if we fail, we are in trouble.
    # But the LUW should work for d=11.
    
    raise ValueError("Failed to construct graph")

# EVOLVE-BLOCK-END

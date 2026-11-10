import random
import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    timeout = 110.0

    # Fallback LUW Construction
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

    def get_girth(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        min_girth = 10**9
        for i in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif parent[u] != v and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
        return min_girth

    # Projective Plane Approach
    def try_projective_plane(k, g):
        # Find smallest prime power >= k
        def is_prime_power(x):
            if x < 2: return False
            for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
                if x % p == 0:
                    y = x
                    while y % p == 0:
                        y //= p
                    if y == 1:
                        return True
            return False
        
        q = k
        while not is_prime_power(q):
            q += 1
            
        # Build Incidence Graph of PG(2, q)
        # Points: (x, y, 1) and (1, y, 0) and (0, 1, 0)
        # Lines: ax + by + cz = 0
        points = []
        lines = []
        
        # Generate points
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
        for y in range(q):
            points.append((1, y, 0))
        points.append((0, 1, 0))
        
        # Generate lines
        for a in range(q):
            for b in range(q):
                lines.append((a, b, 1))
        for b in range(q):
            lines.append((1, b, 0))
        lines.append((0, 1, 0))
        
        n_base = len(points) + len(lines)
        base_edges = []
        # Incidence: point (x,y,z) on line (a,b,c) if ax+by+cz=0
        for i, p in enumerate(points):
            for j, l in enumerate(lines):
                if p[0]*l[0] + p[1]*l[1] + p[2]*l[2] == 0:
                    base_edges.append((i, n_base//2 + j))
                    
        # base_edges is a (q+1)-regular graph
        # Try to lift it
        # We want a k-regular graph.
        # If we take a (q+1)-regular graph, we can remove a matching to get k-regular if q+1 = k+1.
        # Here q=5, k=5. q+1=6.
        # So we need a 6-regular graph, then remove a perfect matching.
        
        # Lift size M
        # Total vertices N = n_base * M
        # We want N < 1116.
        # n_base = 31 + 31 = 62.
        # 62 * M < 1116 => M <= 17.
        
        for M in [10, 12, 15, 17]:
            if time.time() - start_time > timeout - 5:
                break
            N = n_base * M
            # Generate random lift
            # For each edge (u, v) in base, assign a permutation offset?
            # Standard random lift: For each vertex u in base, assign a random permutation \pi_u of Z_M.
            # Edge (u, v) in base becomes edges (u, m) -- (v, \pi_v^{-1}(\pi_u(m)))?
            # Simpler: For each edge (u, v) in base, pick a random shift s in Z_M.
            # Edge becomes (u, m) -- (v, m+s).
            
            lift_edges = []
            adj = [[] for _ in range(N)]
            
            # We need to ensure the lift is simple.
            # With random shifts, multi-edges are possible if multiple base edges map to the same pair.
            # Base is simple. Lift is simple if no two edges (u,v) and (u',v') map to the same (U,V).
            # Since base is simple, (u,v) != (u',v').
            # Collision if u=u', v=v' (impossible) or u=v', v=u' (symmetric).
            # In undirected graph, (u,v) is same as (v,u).
            # Base edges are unique.
            # So lift is simple if we handle the undirected nature correctly.
            
            for u, v in base_edges:
                s = random.randint(0, M - 1)
                for m in range(M):
                    u_lift = u * M + m
                    v_lift = v * M + (m + s) % M
                    if u_lift != v_lift:
                        # Check for duplicate
                        # Since base is simple, and we process each base edge once,
                        # and shifts are fixed per edge,
                        # we might create multi-edges if (u,v) and (v,u) are both in base_edges?
                        # No, base_edges contains each undirected edge once.
                        # So this should be simple.
                        lift_edges.append((u_lift, v_lift))
                        adj[u_lift].append(v_lift)
                        adj[v_lift].append(u_lift)
                        
            # Check Girth
            girth = get_girth(N, lift_edges)
            if girth >= g:
                # Now reduce degree from 6 to 5
                # Remove a perfect matching
                # Find a perfect matching in the 6-regular graph
                # Greedy matching
                matched = [False] * N
                matching = []
                for u in range(N):
                    if not matched[u]:
                        # Find an unmatched neighbor
                        # This is not a perfect matching algorithm, just greedy.
                        # Might fail.
                        pass 
                
                # Better: Randomly remove edges to reduce degree to 5.
                # This might break girth? No, removing edges preserves girth lower bound.
                # But we need *exactly* degree 5.
                # We can just randomly remove 1 edge per vertex? No, must be a matching.
                
                # If we just return the 6-regular graph, it's invalid (degree k=5).
                # So we must reduce.
                
                # Simple reduction: 
                # For each vertex, pick one neighbor to remove.
                # Ensure it's a matching.
                
                # This is complex. 
                # If girth is good, we can try to find a matching.
                
                # For now, if we find a 6-regular graph with girth >= 9, 
                # we can try to remove a random perfect matching.
                
                # Generate random permutation for matching?
                # Just return the 6-regular graph? No, invalid.
                
                # Fallback to LUW if this is too complex.
                pass
                
    # Use LUW
    return luw_construct(k, g)

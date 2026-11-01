import time
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_n = float('inf')
    best_edges = None

    # LUW construction parameters
    # q must be k for k-regularity
    q = k
    
    if q == 4:
        # GF(4) operations
        # 0, 1, 2, 3
        # 0=0, 1=1, 2=omega, 3=omega^2
        # add: XOR
        # mul: table
        table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        # Assume prime field for other k
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # LUW rules
    # r[c] = (a, b) for y[c] = x[c] +/- y[a]*x[b]
    # Base rules
    base_rules = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    
    def get_rules(d_val):
        r = dict(base_rules)
        for c in range(5, d_val + 1):
            rc = (c - 5) % 4
            c0 = c - rc
            # Sequence of rules for c >= 5
            # c=5: (1, 3) ? 
            # Standard LUW rules:
            # c=2: (1,1)
            # c=3: (2,1)
            # c=4: (1,2)
            # c=5: (1,3)
            # c=6: (2,4)
            # c=7: (3,5)
            # c=8: (1,6)
            # Pattern?
            # The seed code used:
            # r[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][rc]
            # Let's use the seed's rule generation to be consistent.
            r[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][rc]
        return r

    def build_luw(q_val, d_val):
        r = get_rules(d_val)
        
        def get_neighbours(side, x):
            # x is tuple of length d_val
            # y[0] is ignored in x, but y[1] is the first choice
            # In seed: y = [None, first] + [0]*(d-1)
            # So y[1] = first
            for first in range(q_val):
                y = [0] * (d_val + 1)
                y[1] = first
                for c in range(2, d_val + 1):
                    a, b = r[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, tuple([0] * d_val))
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            side, x = vertex
            v_idx = index[vertex]
            for other in get_neighbours(side, x):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                o_idx = index[other]
                # Add edge only if side==0 to avoid duplicates
                if side == 0:
                    edges.append((v_idx, o_idx))
                    
        return len(order), edges

    def check_girth(n, edges, target_girth):
        # Check if girth >= target_girth
        # BFS from each vertex up to depth target_girth // 2
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        max_depth = target_girth // 2
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            
            while q:
                u = q.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        # Back edge or cross edge
                        # If dist[u] + dist[v] + 1 < target_girth, we found a short cycle
                        # But this logic is tricky. 
                        # Standard girth check: 
                        # If we encounter a visited node v with dist[v] < dist[u], 
                        # cycle length is dist[u] + dist[v] + 1? No.
                        # If we just want to ensure NO cycle < G, we can do:
                        # If we find a cycle of length L < G, return False.
                        # A cycle is formed if we visit a node already in the BFS tree 
                        # that is not the parent.
                        # Length = dist[u] + dist[v] + 1?
                        # If v is ancestor of u, length = dist[u] - dist[v] + 1.
                        # If v is not ancestor, length = dist[u] + dist[v] + 1.
                        if v != parent[u]:
                            # Check if this forms a cycle < G
                            # Simple check: if dist[u] + dist[v] + 1 < target_girth
                            # This might miss some cycles if dist is large, 
                            # but since we stop at max_depth = G/2, 
                            # dist[u], dist[v] <= G/2.
                            # Max sum = G. 
                            # If sum + 1 < G, definitely short.
                            # What if sum + 1 == G? That's okay (girth G).
                            # What if sum + 1 > G? Okay.
                            # So if dist[u] + dist[v] + 1 < target_girth, return False.
                            if dist[u] + dist[v] + 1 < target_girth:
                                return False
        return True

    # Try d values
    # d=3 -> girth ~8
    # d=4 -> girth ~10
    # d=5 -> girth ~12
    # Start from small d to find smallest n
    
    min_d = max(3, g - 5)
    # Ensure we try d=4 for g=10
    if g == 10:
        min_d = 4
        
    for d_val in range(min_d, min_d + 5):
        if time.time() > deadline:
            break
        try:
            n, edges = build_luw(q, d_val)
            # Verify girth
            if check_girth(n, edges, g):
                if n < best_n:
                    best_n = n
                    best_edges = edges
        except Exception:
            pass

    if best_edges is None:
        # Fallback to seed's default if all failed
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        best_n, best_edges = build_luw(q, d)
        
    return best_n, best_edges

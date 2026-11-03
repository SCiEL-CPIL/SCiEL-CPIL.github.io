import time
import sys
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = 115.0  # seconds; keep margin before 150s kill

    # The LUW construction works for q = k if k is a prime power.
    # For k=3, q=3. For k=5, q=5, etc.
    # If k is not a prime power, this method fails, but the task specifies k=3.
    q = k
    
    # Precompute field operations for efficiency
    if q == 3:
        add_tbl = [[0,1,2],[1,2,0],[2,0,1]]
        sub_tbl = [[0,2,1],[1,0,2],[2,1,0]]
        mul_tbl = [[0,0,0],[0,1,2],[0,2,1]]
    elif q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add_tbl = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
        sub_tbl = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
        mul_tbl = table
    else:
        # General prime field
        add_tbl = [[(i+j)%q for j in range(q)] for i in range(q)]
        sub_tbl = [[(i-j)%q for j in range(q)] for i in range(q)]
        mul_tbl = [[(i*j)%q for j in range(q)] for i in range(q)]

    def build_rule(d):
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        return rule

    def build_graph(d, rule):
        # Precompute rule arrays for faster access
        rule_a = [0]*(d+1)
        rule_b = [0]*(d+1)
        for c in range(2, d+1):
            rule_a[c], rule_b[c] = rule[c]
        
        def neighbours(vertex):
            side, x = vertex
            if side == 0:
                for first in range(q):
                    y = [0]*(d+1)
                    y[1] = first
                    for c in range(2, d+1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = add_tbl[x[c]][mul_tbl[y[a]][x[b]]]
                    yield 1, tuple(y)
            else:
                for first in range(q):
                    y = [0]*(d+1)
                    y[1] = first
                    for c in range(2, d+1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = sub_tbl[x[c]][mul_tbl[x[a]][y[b]]]
                    yield 0, tuple(y)

        start = (0, (0,)*(d+1))
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

        n = len(order)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return n, edges, adj

    def verify_girth(n, adj, min_girth):
        """
        Check girth >= min_girth using BFS from each vertex.
        """
        max_dist = (min_girth - 1) // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= max_dist:
                    continue
                pu = parent[u]
                for v in adj[u]:
                    if v == pu:
                        continue
                    dv = dist[v]
                    if dv == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    elif dv >= du:
                        cycle_len = du + dv + 1
                        if cycle_len < min_girth:
                            return False
        return True

    # Theoretical minimum d for girth g is ceil((g-1)/2)
    d_min = (g + 1) // 2
    
    # Try increasing d until we find a valid graph or run out of time
    max_d = min(15, (int(1e6) // (2 * (q ** d_min))) + d_min)  # Cap based on memory
    max_d = max(max_d, d_min)
    
    for d in range(d_min, max_d + 1):
        if time.time() - t0 > deadline - 5.0:
            break
            
        rule = build_rule(d)
        try:
            n, edges, adj = build_graph(d, rule)
        except MemoryError:
            break
        except Exception:
            continue

        # Check girth
        if verify_girth(n, adj, g):
            return n, edges
        
    # Fallback: return last constructed graph if valid, else empty
    # This should not happen for k=3, g=16 with d_min=8
    return 0, []

# EVOLVE-BLOCK-END

import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Fallback: LUW Construction
    def luw_construct(k, g):
        q = k
        # d such that girth >= g
        # Girth of CD(d,q) is 2d+4? Prompt says d+5.
        # Let's assume we need 2d+4 >= g or d+5 >= g.
        # For g=9, d=4 gives 12 or 9.
        # Let's use the prompt's logic: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
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
        return len(order), edges

    # 2. Voltage Graph Construction over K_{k,k}
    def voltage_graph_construct(k, g):
        """
        Construct a k-regular graph using voltage graph over K_{k,k}.
        
        The base graph is K_{k,k} with bipartition (L, R) where |L| = |R| = k.
        Each edge (i, j) between L_i and R_j has a voltage in Z_m.
        The lift has vertices {(i, a) : i in L, a in Z_m} and {(j, b) : j in R, b in Z_m}.
        Total vertices: 2*k*m.
        
        An edge exists between (i, a) and (j, b) if and only if b = a + voltage(i,j) mod m.
        
        To ensure girth >= g, we need to avoid short cycles.
        The shortest cycles in K_{k,k} are 4-cycles: L_i - R_j - L_p - R_q - L_i.
        The voltage sum around such a cycle is:
        v(i,j) + v(p,q) - v(i,q) - v(p,j) mod m
        
        If this is non-zero for all 4-cycles, then there are no 4-cycles in the lift.
        We also need to check 6-cycles, 8-cycles, etc.
        """
        
        # Try different values of m
        # n = 2*k*m
        # We want n to be as small as possible, so try small m first.
        # But m must be large enough to avoid short cycles.
        
        # For k=5, g=9, let's try m from 50 to 200
        # n ranges from 500 to 2000
        
        # Generate random voltage assignments and check girth
        best_n = None
        best_edges = None
        
        # Use deterministic seed for reproducibility
        rng = random.Random(42)
        
        for m in range(50, 201, 10):
            if time.time() > deadline:
                break
            
            n = 2 * k * m
            # Try several random voltage assignments
            for attempt in range(10):
                if time.time() > deadline:
                    break
                
                # Generate random voltages
                # voltages[i][j] = voltage of edge from L_i to R_j
                voltages = [[rng.randint(0, m-1) for _ in range(k)] for _ in range(k)]
                
                # Check if this voltage assignment gives girth >= g
                # Build the graph and check girth
                edges = []
                adj = [[] for _ in range(n)]
                
                for i in range(k):
                    for j in range(k):
                        v = voltages[i][j]
                        for a in range(m):
                            b = (a + v) % m
                            u = i * m + a  # Vertex in L_i
                            w = k * m + j * m + b  # Vertex in R_j
                            edges.append((u, w))
                            adj[u].append(w)
                            adj[w].append(u)
                
                # Check girth using BFS
                def check_girth(adj, n, target_g):
                    min_girth = float('inf')
                    for start in range(n):
                        dist = [-1] * n
                        parent = [-1] * n
                        dist[start] = 0
                        q = deque([start])
                        while q:
                            u = q.popleft()
                            for w in adj[u]:
                                if dist[w] == -1:
                                    dist[w] = dist[u] + 1
                                    parent[w] = u
                                    q.append(w)
                                elif w != start and parent[w] != u and parent[u] != w:
                                    cycle_len = dist[u] + dist[w] + 1
                                    if cycle_len < min_girth:
                                        min_girth = cycle_len
                                    if min_girth < target_g:
                                        return min_girth
                    return min_girth
                
                girth = check_girth(adj, n, g)
                
                if girth >= g:
                    # Found a valid graph
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges
                        # If we found a good one, we can break early
                        if n < 1000:
                            return best_n, best_edges
        
        return best_n, best_edges

    # Try voltage graph construction first
    n_vg, edges_vg = voltage_graph_construct(k, g)
    
    if n_vg is not None and edges_vg is not None:
        # Verify the graph
        # Check regularity
        deg = [0] * n_vg
        for u, v in edges_vg:
            deg[u] += 1
            deg[v] += 1
        if all(d == k for d in deg):
            return n_vg, edges_vg
    
    # Fallback to LUW
    n_luw, edges_luw = luw_construct(k, g)
    return n_luw, edges_luw

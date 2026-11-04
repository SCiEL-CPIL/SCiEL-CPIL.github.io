import time
from collections import deque
import numpy as np
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    if k == 3 and g == 16:
        # Try to construct the Folkman graph (936 vertices)
        # Using voltage graph over K3,3 with Z_156
        n_base = 6
        group_size = 156
        n_total = n_base * group_size
        
        # Base graph K3,3
        # U = {0, 1, 2}, V = {3, 4, 5}
        # Edges: (0,3), (0,4), (0,5), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5)
        base_edges = [(0,3), (0,4), (0,5), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5)]
        
        # Try a few random voltage assignments
        best_n = float('inf')
        best_edges = None
        
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
                
            random.seed(attempt)
            voltages = {}
            for u, v in base_edges:
                voltages[(u, v)] = random.randint(1, group_size - 1)
                # Inverse voltage for (v, u)
                voltages[(v, u)] = (-voltages[(u, v)]) % group_size
            
            # Build the lifted graph
            adj = [[] for _ in range(n_total)]
            edges = []
            edge_set = set()
            
            for u, v in base_edges:
                vu = voltages[(u, v)]
                for i in range(group_size):
                    u_lift = u * group_size + i
                    v_lift = v * group_size + ((i + vu) % group_size)
                    if u_lift < v_lift:
                        if (u_lift, v_lift) not in edge_set:
                            edge_set.add((u_lift, v_lift))
                            adj[u_lift].append(v_lift)
                            adj[v_lift].append(u_lift)
                            edges.append((u_lift, v_lift))
                    else:
                        if (v_lift, u_lift) not in edge_set:
                            edge_set.add((v_lift, u_lift))
                            adj[v_lift].append(u_lift)
                            adj[u_lift].append(v_lift)
                            edges.append((v_lift, u_lift))
            
            # Check girth
            if _check_girth_fast(adj, n_total, g):
                best_n = n_total
                best_edges = edges
                break
        
        if best_edges is not None:
            return best_n, best_edges
        
        # Fallback to LUW
        n_luw, edges_luw = _luw_graph(k, g)
        return n_luw, edges_luw
    
    # General fallback
    n_luw, edges_luw = _luw_graph(k, g)
    return n_luw, edges_luw

def _check_girth_fast(adj, n, g_limit):
    max_depth = (g_limit - 1) // 2
    for start in range(n):
        dist = np.full(n, -1, dtype=np.int32)
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] < max_depth:
                        queue.append(v)
                else:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g_limit:
                        return False
    return True

def _luw_graph(k, g):
    # Same as before
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

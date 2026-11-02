import time
import random
import numpy as np

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    if k == 4 and g == 10:
        # Try Voltage Graph Construction on K5 with Z8 x Z8
        # Base graph K5
        base_vertices = [0, 1, 2, 3, 4]
        base_edges = [(i, j) for i in range(5) for j in range(i+1, 5)]
        n_base = 5
        m = 8
        l = 8
        n_lift = n_base * m * l
        
        def check_girth(edges, n):
            # Build adjacency list
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            min_girth = 10**9
            for start in range(n):
                # BFS to find shortest cycle containing start
                # We can use a standard BFS that detects back edges
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = [start]
                while queue:
                    u = queue.pop(0)
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif v != parent[u]:
                            # Found a cycle
                            # The cycle length is dist[u] + dist[v] + 1
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
                # Optimization: if min_girth is already small, we can break early?
                # But we need to know if it's >= 10.
                # If we find a cycle < 10, we can return False immediately.
                if min_girth < 10:
                    return min_girth
            return min_girth

        def build_lift(voltages):
            # voltages is a dict {(u,v): (a,b)} for u < v
            edges = []
            for u, v in base_edges:
                a, b = voltages[(u,v)]
                for i in range(m):
                    for j in range(l):
                        g1 = (i, j)
                        # u -> v
                        g2 = ((i + a) % m, (j + b) % l)
                        u_node = u * (m*l) + i * l + j
                        v_node = v * (m*l) + g2[0] * l + g2[1]
                        edges.append((u_node, v_node))
            return edges

        best_edges = None
        best_girth = 0
        
        # Random search
        random.seed(42)
        while time.time() < deadline:
            # Assign random voltages
            voltages = {}
            for u, v in base_edges:
                voltages[(u,v)] = (random.randint(0, m-1), random.randint(0, l-1))
            
            edges = build_lift(voltages)
            
            # Check girth
            # We can do a faster check: just look for cycles < 10
            # If we find one, discard. If not, it's valid.
            # The check_girth function above returns the min girth.
            # We can optimize check_girth to stop as soon as it finds a cycle < 10.
            
            girth = check_girth(edges, n_lift)
            
            if girth >= 10:
                return n_lift, edges
            
            if girth > best_girth:
                best_girth = girth
                best_edges = edges
                
        # If no valid graph found, return the best one found (might be invalid if < 10)
        # But the task requires valid graph.
        # If we didn't find one, we must fall back to a known valid construction.
        # The LUW construction is valid.
        
    # Fallback: LUW Construction from seed
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

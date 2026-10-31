import random
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Fallback: L-U-W graph (Seed)
    def l_u_w_graph(k, g):
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

    # Try to find a smaller graph using voltage lifts of Petersen
    # Petersen graph: 10 vertices, 15 edges
    petersen_edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                      (1, 6), (1, 7), (1, 8), (1, 9), (1, 2),
                      (2, 3), (2, 4), (2, 5), (2, 6),
                      (3, 7), (3, 8), (3, 9),
                      (4, 5), (4, 6), (4, 7),
                      (5, 8), (5, 9),
                      (6, 7), (6, 8),
                      (7, 9),
                      (8, 9)]
    # Actually, standard Petersen edges:
    # Outer 5-cycle: 0-1-2-3-4-0
    # Inner 5-star: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
                      (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
                      (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
    
    def get_lift_edges(base_edges, voltages, m):
        # base_edges: list of (u, v)
        # voltages: dict (u, v) -> voltage
        # Returns edges of the lift
        edges = []
        for u, v in base_edges:
            volt = voltages.get((u, v), voltages.get((v, u), 0))
            # Edge (u, i) to (v, (i + volt) % m)
            for i in range(m):
                edges.append((u * m + i, v * m + (i + volt) % m))
            # Undirected, so also (v, (i - volt) % m) to (u, i)
            # But we just add both directions in the undirected sense
            # Actually, for undirected graph, we add (u, i) - (v, (i+volt)%m)
            # and (v, (i-volt)%m) - (u, i) is the same edge.
            # So we just add one edge per i.
        return edges

    def check_girth(n, edges, target_girth):
        # BFS from each vertex
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                if dist[u] >= target_girth // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Cycle found
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < target_girth:
                            return False
        return True

    best_n, best_edges = None, None
    # Try small lifts
    for m in [50, 100, 200, 500, 1000]:
        if time.time() > deadline:
            break
        n = 10 * m
        # Random voltages
        for _ in range(10):
            if time.time() > deadline:
                break
            voltages = {}
            for u, v in petersen_edges:
                voltages[(u, v)] = random.randint(0, m - 1)
                voltages[(v, u)] = (-voltages[(u, v)]) % m
            
            edges = get_lift_edges(petersen_edges, voltages, m)
            # Check if simple (no multi-edges)
            # For Petersen lift, multi-edges can occur if voltages are 0 or m/2
            # We assume m is large enough and random voltages avoid this mostly
            # Verify girth
            if check_girth(n, edges, g):
                best_n, best_edges = n, edges
                break
        if best_n is not None:
            break

    if best_n is not None:
        return best_n, best_edges
    
    # Fallback to L-U-W
    return l_u_w_graph(k, g)

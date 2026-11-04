"""Cage graph constructor: 3-regular, girth >= g, minimize n."""
import time
import random
import sys

_START = time.time()
_DEADLINE = _START + 140  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _deadline_hit():
    return _elapsed() > _DEADLINE


def construct(k, g):
    """Construct k-regular graph with girth >= g on as few vertices as possible."""
    if k == 3 and g == 16:
        # Use voltage lift of Petersen graph
        # Petersen has 10 vertices, 15 edges.
        # Lift by Z_m gives 10*m vertices.
        # Target n ~ 512. m ~ 51.
        
        # Define Petersen graph edges
        # Outer cycle: 0-1-2-3-4-0
        # Inner star: 5-7-9-6-8-5
        # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        petersen_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        petersen_adj = [[] for _ in range(10)]
        for u, v in petersen_edges:
            petersen_adj[u].append(v)
            petersen_adj[v].append(u)

        def check_girth_lift(m, voltages):
            """
            Check if the lift has girth >= 16.
            voltages: dict edge_index -> voltage in Z_m
            """
            # Build adjacency of the lift
            # Vertex (base_v, layer) -> (base_v + 0..m-1, layer)
            # Total vertices 10*m
            n = 10 * m
            adj = [[] for _ in range(n)]
            
            for i, (u, v) in enumerate(petersen_edges):
                vol = voltages[i]
                for l in range(m):
                    u_l = u * m + l
                    v_l = v * m + (l + vol) % m
                    adj[u_l].append(v_l)
                    adj[v_l].append(u_l)
            
            # Check girth >= 16 using BFS from each vertex
            # Optimized BFS: stop if depth >= 8 (half of 16)
            # Since graph is bipartite, cycles are even.
            # We just need to ensure no cycle of length < 16.
            # BFS from each node, tracking distance.
            # If we find a neighbor at distance d, cycle length is d + dist[neighbor] + 1?
            # No, standard BFS for girth.
            
            for start in range(n):
                if _deadline_hit():
                    return False
                dist = [-1] * n
                dist[start] = 0
                queue = [start]
                head = 0
                while head < len(queue):
                    u = queue[head]
                    head += 1
                    du = dist[u]
                    if du >= 8: # max depth 8, so cycle len <= 16
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = du + 1
                            queue.append(v)
                        else:
                            # Found a cycle
                            # Cycle length is du + dist[v] + 1
                            # But we need to be careful: this might be the tree edge
                            # If v is parent of u, skip.
                            # In BFS, if dist[v] < dist[u], v is parent or ancestor.
                            # If dist[v] == dist[u], it's a cross edge in same level? No, bipartite.
                            # In bipartite graph, edges only go between levels.
                            # So if u is at level du, v is at level du+1 or du-1.
                            # If dist[v] == du+1, it's a forward edge (tree).
                            # If dist[v] == du-1, it's a back edge (parent).
                            # If dist[v] < du-1, it's a cycle.
                            # If dist[v] > du+1, it's a cycle? No, BFS ensures dist[v] <= du+1 for neighbors.
                            
                            # Actually, simpler:
                            # If dist[v] != -1 and v != parent(u), we found a cycle.
                            # We need parent tracking.
                            
                            # Let's use a standard girth check.
                            pass
            # The above BFS is flawed for girth without parent tracking.
            # Let's use a correct but slightly slower BFS.
            
            for start in range(n):
                if _deadline_hit():
                    return False
                visited = {start: (-1, 0)}
                queue = [(start, -1, 0)]
                while queue:
                    u, parent, depth = queue.pop(0)
                    if depth >= 8:
                        continue
                    for v in adj[u]:
                        if v == parent:
                            continue
                        if v in visited:
                            cycle_len = depth + visited[v][1] + 1
                            if cycle_len < 16:
                                return False
                        else:
                            visited[v] = (u, depth + 1)
                            queue.append((v, u, depth + 1))
            return True

        # Try to find voltages for m=51, 52, etc.
        random.seed(42)
        for m in range(51, 60):
            if _deadline_hit():
                break
            for attempt in range(50):
                if _deadline_hit():
                    break
                voltages = [random.randint(0, m - 1) for _ in range(15)]
                if check_girth_lift(m, voltages):
                    # Construct the graph
                    n = 10 * m
                    edges = []
                    for i, (u, v) in enumerate(petersen_edges):
                        vol = voltages[i]
                        for l in range(m):
                            u_l = u * m + l
                            v_l = v * m + (l + vol) % m
                            edges.append((u_l, v_l))
                    return n, edges

        # Fallback to LUW or random
        # Use the seed's fallback if this fails
        q = 3
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
                    y[c] = (x[c] + y[a] * x[b]) % q if side == 0 else (x[c] - x[a] * y[b]) % q
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

    # Generic fallback
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

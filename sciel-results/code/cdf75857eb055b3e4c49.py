import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Fallback: LUW Construction
    def get_luw():
        q = k
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for i, vertex in enumerate(order):
            if time.time() > deadline:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # 2. Search for smaller random graph
    def check_girth(n, edges, g_target):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        max_depth = g_target // 2
        for start in range(n):
            # BFS
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif v != start and dist[v] <= dist[u]: # Cycle found
                        # Check if cycle length < g_target
                        # This simple check is not enough, need to verify
                        # Actually, if we find a back edge to a node with dist <= dist[u], 
                        # the cycle length is dist[u] + dist[v] + 1.
                        # If dist[u] + dist[v] + 1 < g_target, fail.
                        if dist[u] + dist[v] + 1 < g_target:
                            return False
            # If no short cycle found from this start, continue
        return True

    def random_3_reg(n):
        if n % 2 != 0:
            return None
        # Configuration model
        half_edges = n * 3 // 2
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
        random.shuffle(stubs)
        edges = []
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None # Loop
            if (u, v) in edges or (v, u) in edges:
                return None # Multi-edge
            edges.append((u, v))
        return edges

    best_n, best_edges = None, None
    # Try LUW first to have a valid fallback
    luw_n, luw_edges = get_luw()
    best_n, best_edges = luw_n, luw_edges
    
    # Try smaller n
    # The Moore bound is ~510.
    # Try a few random graphs
    for n_try in range(512, 1500, 2):
        if time.time() > deadline:
            break
        if n_try >= best_n:
            break
        for _ in range(10): # 10 attempts per n
            if time.time() > deadline:
                break
            edges = random_3_reg(n_try)
            if edges is None:
                continue
            if check_girth(n_try, edges, g):
                best_n, best_edges = n_try, edges
                break # Found one, try even smaller? No, just return this one?
                # Actually, if we find one, we can break and return it.
                # But maybe a smaller one exists?
                # We are iterating n_try upwards. So we want the smallest.
                # So if we find one at n_try, we can return it immediately?
                # Yes, because we want to minimize n.
                return n_try, edges

    return best_n, best_edges
# EVOLVE-BLOCK-END

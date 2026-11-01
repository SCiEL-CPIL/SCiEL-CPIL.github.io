import time
import random
import numpy as np
from collections import deque

_DEADLINE = 110.0

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
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

    if k != 4 or g != 10:
        return luw_construct(k, g)

    n = 320
    t0 = time.time()

    # Generate a random 4-regular graph using configuration model
    def gen_random_4reg(n):
        while True:
            stubs = list(range(n * 4))
            random.shuffle(stubs)
            edges = []
            valid = True
            seen = set()
            for i in range(0, len(stubs), 2):
                u = stubs[i] // 4
                v = stubs[i + 1] // 4
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in seen:
                    valid = False
                    break
                seen.add(key)
                edges.append(key)
            if valid:
                return edges

    edges = gen_random_4reg(n)

    # Build adjacency list for fast access
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        adj[i].sort()

    # Fast girth check using BFS from each vertex
    # Returns the minimum cycle length found (or infinity if none < g)
    # Also returns a "cost" = sum of (g - cycle_len) for all cycles found
    def check_girth(adj, n, g):
        """Check if graph has girth >= g. Returns (min_cycle, cost, problem_vertices)."""
        min_cycle = g
        cost = 0
        problem_vertices = set()
        half_g = g // 2  # 5 for g=10

        for start in range(n):
            # BFS from start, tracking parent
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])

            while queue:
                u = queue.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif dist[v] > dist[u]:
                        # Back edge: cycle length = dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            min_cycle = min(min_cycle, cycle_len)
                            cost += (g - cycle_len)
                            problem_vertices.add(u)
                            problem_vertices.add(v)

        return min_cycle, cost, problem_vertices

    # Initial check
    min_c, current_cost, prob_verts = check_girth(adj, n, g)

    if current_cost == 0:
        return n, edges

    best_cost = current_cost
    best_edges = edges[:]
    best_adj = [lst[:] for lst in adj]

    T = 2.0
    cooling = 0.9995

    def do_swap(u, v, x, y, adj):
        """Swap edges (u,v) and (x,y) with (u,x) and (v,y)."""
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        adj[u].append(x)
        adj[x].append(u)
        adj[v].append(y)
        adj[y].append(v)
        adj[u].sort()
        adj[v].sort()
        adj[x].sort()
        adj[y].sort()

    def undo_swap(u, v, x, y, adj):
        """Undo the swap: restore (u,v),(x,y) from (u,x),(v,y)."""
        adj[u].remove(x)
        adj[x].remove(u)
        adj[v].remove(y)
        adj[y].remove(v)
        adj[u].append(v)
        adj[v].append(u)
        adj[x].append(y)
        adj[y].append(x)
        adj[u].sort()
        adj[v].sort()
        adj[x].sort()
        adj[y].sort()

    # Precompute all edges for faster move generation
    def get_edge_set(adj, n):
        es = set()
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    es.add((i, j))
        return es

    while time.time() - t0 < _DEADLINE:
        if current_cost == 0:
            final_edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        final_edges.append((i, j))
            return n, final_edges

        # Generate a move: pick two edges to swap
        # Strategy: bias toward edges involving problem vertices
        moved = False
        max_attempts = 50
        for _ in range(max_attempts):
            # Pick u from problem vertices if available, else random
            if prob_verts and random.random() < 0.7:
                u = random.choice(list(prob_verts))
            else:
                u = random.randint(0, n - 1)

            # Pick v from u's neighbors
            if not adj[u]:
                continue
            v = random.choice(adj[u])

            # Pick x from problem vertices if available, else random
            if prob_verts and random.random() < 0.7:
                x = random.choice(list(prob_verts))
            else:
                x = random.randint(0, n - 1)

            # Pick y from x's neighbors
            if not adj[x]:
                continue
            y = random.choice(adj[x])

            # Check validity
            if u == v or x == y:
                continue
            if len({u, v, x, y}) < 4:
                continue
            # Check (u,x) and (v,y) are not already edges
            if x in adj[u] or y in adj[v]:
                continue

            # Perform swap
            do_swap(u, v, x, y, adj)
            _, new_cost, new_prob = check_girth(adj, n, g)

            if new_cost < current_cost:
                current_cost = new_cost
                prob_verts = new_prob
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = [lst[:] for lst in adj]
                moved = True
            else:
                # SA acceptance
                delta_c = new_cost - current_cost
                if delta_c == 0:
                    # Accept neutral moves sometimes
                    if random.random() < 0.3:
                        current_cost = new_cost
                        prob_verts = new_prob
                        moved = True
                    else:
                        undo_swap(u, v, x, y, adj)
                elif random.random() < np.exp(-delta_c / T):
                    current_cost = new_cost
                    prob_verts = new_prob
                    moved = True
                else:
                    undo_swap(u, v, x, y, adj)

            if moved:
                break

        T *= cooling
        if T < 0.05:
            T = 0.05

    # If we didn't find a valid graph, return LUW fallback
    return luw_construct(k, g)

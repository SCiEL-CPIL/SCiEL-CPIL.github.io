import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    def luw_construction():
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

    def has_short_cycle(adj, g):
        n = len(adj)
        for s in range(n):
            # BFS from s, depth up to g//2
            # State: (node, parent)
            visited = set()
            stack = [(s, -1, 0)]
            while stack:
                node, parent, depth = stack.pop()
                if depth >= g // 2:
                    continue
                for neighbor in adj[node]:
                    if neighbor == parent:
                        continue
                    if neighbor == s and depth + 1 < g:
                        return True
                    state = (neighbor, node)
                    if state not in visited:
                        visited.add(state)
                        stack.append((neighbor, node, depth + 1))
        return False

    def find_short_cycle(adj, g):
        n = len(adj)
        for s in range(n):
            visited = {}
            stack = [(s, -1, 0)]
            while stack:
                node, parent, depth = stack.pop()
                if depth >= g // 2:
                    continue
                for neighbor in adj[node]:
                    if neighbor == parent:
                        continue
                    if neighbor == s and depth + 1 < g:
                        # Reconstruct cycle
                        # This is tricky with stack, so we just return True/False for now
                        # For local search, we need the cycle.
                        pass
                    state = (neighbor, node)
                    if state not in visited:
                        visited[state] = (node, parent, depth)
                        stack.append((neighbor, node, depth + 1))
        return None

    def random_k_regular(n, k, seed=None):
        if n * k % 2 != 0:
            return None
        rng = random.Random(seed)
        half_edges = list(range(n)) * k
        rng.shuffle(half_edges)
        edges = set()
        adj = [[] for _ in range(n)]
        for i in range(0, n * k, 2):
            u = half_edges[i]
            v = half_edges[i + 1]
            if u == v:
                return None
            key = (min(u, v), max(u, v))
            if key in edges:
                return None
            edges.add(key)
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def local_search(n, k, g, deadline):
        best_adj = None
        for attempt in range(1000):
            if time.time() > deadline - 5:
                break
            adj = random_k_regular(n, k, seed=attempt)
            if adj is None:
                continue
            
            # Check girth
            if not has_short_cycle(adj, g):
                return adj
            
            # Local improvement
            # Try to break short cycles
            # This part is heuristic and might be slow
            # For now, just return if found in random
            pass
            
        return None

    # First, get the LUW construction as baseline
    n_luw, edges_luw = luw_construction()
    best_n, best_edges = n_luw, edges_luw

    if k == 4 and g == 10:
        # Try to find the 320-vertex cage
        target_n = 320
        adj_found = local_search(target_n, k, g, deadline)
        if adj_found:
            # Convert adj to edges
            edges = []
            for i in range(target_n):
                for j in adj_found[i]:
                    if i < j:
                        edges.append((i, j))
            return target_n, edges

    return best_n, best_edges

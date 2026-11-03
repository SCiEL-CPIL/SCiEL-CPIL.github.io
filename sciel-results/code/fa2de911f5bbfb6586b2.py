import random
import time
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    def check_girth_np(adj_np, n, g, start_vertex=0):
        """
        Check girth using non-backtracking BFS from start_vertex with NumPy adjacency.
        Returns True if no cycle shorter than g is found.
        """
        for first_neighbor in np.where(adj_np[start_vertex])[0]:
            visited = np.zeros((n, n), dtype=bool)
            queue = deque()
            queue.append((first_neighbor, start_vertex, 1))
            visited[first_neighbor, start_vertex] = True

            while queue:
                curr, prev, depth = queue.popleft()

                if start_vertex != prev:
                    if adj_np[curr, start_vertex]:
                        cycle_len = depth + 1
                        if cycle_len < g:
                            return False

                if depth + 1 >= g:
                    continue

                neighbors = np.where(adj_np[curr])[0]
                for nxt in neighbors:
                    if nxt == prev:
                        continue
                    if not visited[nxt, curr]:
                        visited[nxt, curr] = True
                        queue.append((nxt, curr, depth + 1))

        return True

    def make_cyclic_graph(n, offsets):
        """
        Create a cyclic Z_n-regular graph with given offsets.
        Returns None if invalid.
        """
        if 0 in offsets:
            return None
        if len(set(o % n for o in offsets)) < len(offsets):
            return None

        for i in range(len(offsets)):
            for j in range(i + 1, len(offsets)):
                if (offsets[i] + offsets[j]) % n == 0:
                    return None

        edges = []
        edge_set = set()

        for i in range(n):
            for off in offsets:
                j = (i + off) % n
                if i == j:
                    return None
                key = (min(i, j), max(i, j))
                if key in edge_set:
                    return None
                edge_set.add(key)
                edges.append((i, j))

        degree = [0] * n
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        if any(d != k for d in degree):
            return None

        return edges

    def check_girth_from_edges_np(edges, n, g):
        adj = np.zeros((n, n), dtype=bool)
        for u, v in edges:
            adj[u, v] = True
            adj[v, u] = True
        return check_girth_np(adj, n, g, 0)

    def random_valid_offsets(n, k, min_offset_frac=0.15):
        """Generate k random valid offsets for a cyclic graph, biased toward larger offsets."""
        min_off = max(1, int(n * min_offset_frac))
        candidates = list(range(min_off, n))
        random.shuffle(candidates)
        
        offsets = []
        for o in candidates:
            if len(offsets) >= k:
                break
            valid = True
            for existing in offsets:
                if (o + existing) % n == 0:
                    valid = False
                    break
            if valid:
                offsets.append(o)
        
        if len(offsets) < k:
            small_candidates = list(range(1, min_off))
            random.shuffle(small_candidates)
            for o in small_candidates:
                if len(offsets) >= k:
                    break
                if o in offsets:
                    continue
                valid = True
                for existing in offsets:
                    if (o + existing) % n == 0:
                        valid = False
                        break
                if valid:
                    offsets.append(o)
        
        if len(offsets) < k:
            return None
        return offsets

    best_n = float('inf')
    best_edges = None

    # Strategy:
    # 1. Try n=320 with many random attempts (cyclic graphs) using fast girth check.
    # 2. If not found, try n=322, 324, etc.
    # 3. If not found, fallback to LW graph.

    search_ns = [320] + list(range(322, 350, 2))

    for n in search_ns:
        if time.time() > deadline - 10:
            break

        if n == 320:
            max_attempts = 200000
        else:
            max_attempts = 20000

        attempts = 0
        while attempts < max_attempts:
            if time.time() > deadline - 5:
                break
            attempts += 1

            offsets = random_valid_offsets(n, k)
            if offsets is None:
                continue

            edges = make_cyclic_graph(n, offsets)
            if edges is not None:
                if check_girth_from_edges_np(edges, n, g):
                    return n, edges

    # Fallback to LW graph
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

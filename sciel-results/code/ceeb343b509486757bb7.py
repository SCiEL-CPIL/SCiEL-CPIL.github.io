import time
import random
from collections import deque
from itertools import combinations
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    def luw_construction():
        """Construct the Lazebnik-Ustimenko-Woldar graph CD(d, q)."""
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

    def check_girth_fast(adj, n, g):
        """Check if graph has girth < g using BFS from each vertex.
        Returns True if girth >= g (valid), False otherwise."""
        # For each vertex, do BFS to depth g-1, checking for cycles
        # A cycle of length < g exists if we can return to start in < g steps
        for s in range(n):
            # BFS from s, tracking (vertex, parent) to avoid backtracking
            visited = {}
            queue = deque()
            for nb in adj[s]:
                visited[(nb, s)] = 1
                queue.append((nb, s))
            while queue:
                v, p = queue.popleft()
                d = visited[(v, p)]
                if d >= g - 1:
                    continue
                for w in adj[v]:
                    if w == p:
                        continue
                    state = (w, v)
                    if state not in visited:
                        visited[state] = d + 1
                        if w == s and v != adj[s][0] and v != adj[s][-1]:
                            # Found a cycle back to s of length d+1
                            if d + 1 < g:
                                return False
                        queue.append(state)
                    elif w == s:
                        if d + 1 < g:
                            return False
        return True

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        return adj

    def random_k_regular_fast(n, k, seed):
        """Generate a random k-regular graph using configuration model with retry."""
        if n * k % 2 != 0:
            return None
        rng = random.Random(seed)
        max_retries = 50
        for _ in range(max_retries):
            half_edges = []
            for i in range(n):
                half_edges.extend([i] * k)
            rng.shuffle(half_edges)
            adj = [[] for _ in range(n)]
            edge_set = set()
            valid = True
            for i in range(0, n * k, 2):
                u = half_edges[i]
                v = half_edges[i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
                adj[u].append(v)
                adj[v].append(u)
            if valid:
                return sorted([list(e) for e in edge_set])
        return None

    def try_find_graph(n_target, k, g, deadline, max_attempts=100):
        """Try to find a valid k-regular graph with girth >= g on n_target vertices."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 3:
                break
            edges = random_k_regular_fast(n_target, k, seed=attempt * 1000 + 42)
            if edges is None:
                continue
            adj = build_adj(edges, n_target)
            if check_girth_fast(adj, n_target, g):
                return n_target, edges
        return None

    # Get LUW baseline
    n_luw, edges_luw = luw_construction()
    best_n, best_edges = n_luw, edges_luw

    # Correct Moore bound: 2 * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(g/2 - 1))
    half_g = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(half_g))
    if moore_bound % 2 != 0:
        moore_bound += 1

    # Search strategy: try sizes from moore_bound up to best_n
    # Focus on sizes that are multiples of 2 and near the expected minimum
    candidates = []
    # Add sizes from moore_bound to best_n in steps of 2
    for n_try in range(moore_bound, best_n + 1, 2):
        candidates.append(n_try)
    
    # Also add some specific candidate sizes near 320
    for n_try in range(300, 321, 2):
        if n_try not in candidates and n_try >= moore_bound:
            candidates.append(n_try)
    
    candidates = sorted(set(candidates))
    
    # Limit candidates to avoid timeout
    max_candidates = 30
    if len(candidates) > max_candidates:
        # Prioritize smaller candidates and those near 320
        def priority(n):
            if 310 <= n <= 320:
                return 0
            elif n < 310:
                return 1
            else:
                return 2
        candidates.sort(key=lambda x: (priority(x), x))
        candidates = candidates[:max_candidates]
    
    for n_try in candidates:
        if time.time() > deadline - 5:
            break
        result = try_find_graph(n_try, k, g, deadline, max_attempts=50)
        if result is not None:
            n_found, edges_found = result
            if n_found < best_n:
                best_n, best_edges = n_found, edges_found
                # Found a smaller one, can break early
                break

    return best_n, best_edges

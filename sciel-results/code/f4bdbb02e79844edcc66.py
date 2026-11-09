import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    def time_left():
        return deadline - time.time()

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def next_prime(n):
        if n <= 2:
            return 2
        c = n if n % 2 else n + 1
        while not is_prime(c):
            c += 2
        return c

    def luw_construct(k, g, q=None):
        if q is None:
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

    def has_short_cycle(n, adj, g):
        """Fast girth check: BFS from each vertex up to depth (g-1)//2."""
        half = (g - 1) // 2
        adj_list = [list(s) for s in adj]
        for start in range(n):
            depth = [-1] * n
            parent = [-1] * n
            depth[start] = 0
            parent[start] = -2
            queue = deque([start])
            while queue:
                node = queue.popleft()
                nd = depth[node]
                if nd >= half:
                    continue
                for nb in adj_list[node]:
                    if nb == parent[node]:
                        continue
                    if depth[nb] == -1:
                        depth[nb] = nd + 1
                        parent[nb] = node
                        queue.append(nb)
                    elif parent[nb] != node and depth[nb] != nd + 1:
                        u, v = node, nb
                        d1, d2 = depth[node], depth[nb]
                        while u != v:
                            if d1 > d2:
                                u = parent[u]
                                d1 -= 1
                            elif d2 > d1:
                                v = parent[v]
                                d2 -= 1
                            else:
                                u = parent[u]
                                v = parent[v]
                                d1 -= 1
                                d2 -= 1
                        cycle_len = depth[node] + depth[nb] - 2 * d1 + 1
                        if cycle_len < g:
                            return True
        return False

    def random_regular_graph(n, k, rng):
        if n * k % 2 != 0:
            return None
        for _ in range(300):
            stubs = [i for i in range(n) for _ in range(k)]
            rng.shuffle(stubs)
            edges = []
            seen = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in seen:
                    valid = False
                    break
                seen.add(key)
                edges.append((u, v))
            if valid:
                return edges
        return None

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def try_edge_swap(adj, n, rng):
        for _ in range(100):
            a = rng.randint(0, n - 1)
            if not adj[a]:
                continue
            b = rng.choice(list(adj[a]))
            c = rng.randint(0, n - 1)
            if c == a or c == b:
                continue
            if not adj[c]:
                continue
            d = rng.choice(list(adj[c]))
            if d == a or d == b:
                continue
            if c in adj[a] or d in adj[b]:
                continue
            adj[a].discard(b)
            adj[b].discard(a)
            adj[c].discard(d)
            adj[d].discard(c)
            adj[a].add(c)
            adj[c].add(a)
            adj[b].add(d)
            adj[d].add(b)
            return True
        return False

    # Try LUW with different field sizes to find the smallest valid graph
    best_n = float('inf')
    best_edges = None

    # Standard LUW with q=k
    n_luw, edges_luw = luw_construct(k, g)
    adj_luw = build_adj(n_luw, edges_luw)
    if not has_short_cycle(n_luw, adj_luw, g):
        best_n = n_luw
        best_edges = edges_luw

    # Try smaller prime field sizes (gives fewer vertices)
    moore_bound = 1 + k * sum((k - 1) ** i for i in range((g - 1) // 2))
    # LUW graph has roughly 2*q*d vertices; try primes that give n < best_n
    max_q_for_best = best_n // (2 * max(3, g - 5 if (g - 5) % 2 else g - 4)) if best_n < float('inf') else 100

    q_candidates = []
    for q_try in range(3, min(max_q_for_best, 50) + 1):
        if is_prime(q_try) and q_try >= k:
            q_candidates.append(q_try)
    # Also try q = k and nearby primes
    if k not in q_candidates:
        q_candidates.append(k)
    # Sort by expected size (smaller q = smaller graph)
    q_candidates = sorted(set(q_candidates))

    for q_try in q_candidates:
        if time_left() < 10:
            break
        if best_n < float('inf') and 2 * q_try * max(3, g - 5 if (g - 5) % 2 else g - 4) >= best_n:
            continue
        try:
            n_try, edges_try = luw_construct(k, g, q=q_try)
            if n_try >= best_n:
                continue
            adj_try = build_adj(n_try, edges_try)
            if not has_short_cycle(n_try, adj_try, g):
                best_n = n_try
                best_edges = edges_try
        except Exception:
            continue

    # If LUW didn't give a valid graph, fall back to random search
    if best_n == float('inf'):
        best_n = n_luw
        best_edges = edges_luw

    # Try random search for smaller graphs
    if best_n > moore_bound and time_left() > 20:
        rng = random.Random(42)
        for n in range(max(moore_bound, k + 1), best_n):
            if time_left() < 15:
                break
            if n * k % 2 != 0:
                continue
            for attempt in range(30):
                if time_left() < 8:
                    break
                edges = random_regular_graph(n, k, rng)
                if edges is None:
                    continue
                adj = build_adj(n, edges)
                if not has_short_cycle(n, adj, g):
                    best_n = n
                    best_edges = edges
                    break
                for _ in range(500):
                    if time_left() < 5:
                        break
                    if not try_edge_swap(adj, n, rng):
                        break
                    if not has_short_cycle(n, adj, g):
                        best_n = n
                        best_edges = [(min(u, v), max(u, v)) for u in range(n) for v in adj[u] if u < v]
                        break
            if best_n <= n:
                break

    return best_n, best_edges

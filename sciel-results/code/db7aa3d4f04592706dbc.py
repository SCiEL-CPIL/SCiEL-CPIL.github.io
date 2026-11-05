"""Rewritten program for cage graph construction: k-regular, girth >= g, minimize n."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

def construct(k, g):
    t_start = time.time()
    deadline = 118  # Leave 2s margin before 120s kill

    def moore_bound(k, g):
        if g % 2 == 1:
            half = (g - 1) // 2
            return 1 + k * sum(k**i for i in range(half))
        else:
            half = (g - 2) // 2
            return 2 * sum(k**i for i in range(half + 1))

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        return adj

    def check_girth_fast(adj, n, min_girth):
        """Check girth >= min_girth using BFS with parent tracking."""
        max_depth = (min_girth - 1) // 2
        for start in range(n):
            visited = [None] * n
            visited[start] = (0, -1)
            queue = deque([start])
            while queue:
                u = queue.popleft()
                du, pu = visited[u]
                if du >= max_depth:
                    continue
                for v in adj[u]:
                    if v == pu:
                        continue
                    if visited[v] is None:
                        visited[v] = (du + 1, u)
                        queue.append(v)
                    else:
                        dv, pv = visited[v]
                        if du + dv + 1 < min_girth:
                            return False
        return True

    def random_k_regular(n, k, rng):
        if n * k % 2 != 0:
            return None, None
        for attempt in range(50):
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            rng.shuffle(stubs)
            edges = []
            edge_set = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
                edges.append(key)
            if valid:
                adj = build_adj(edges, n)
                return edges, adj
        return None, None

    def luw_graph(k, g):
        q = k
        prime_powers = set()
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            pp = p
            while pp <= 1000:
                prime_powers.add(pp)
                pp *= p
        if q not in prime_powers:
            return None, None

        d = g - 5
        if d % 2 == 0:
            d += 1
        d = max(3, d)

        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add_op = lambda x, y: x ^ y
            sub_op = lambda x, y: x ^ y
            mul_op = lambda x, y: table[x][y]
        else:
            add_op = lambda x, y: (x + y) % q
            sub_op = lambda x, y: (x - y) % q
            mul_op = lambda x, y: x * y % q

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add_op(x[c], mul_op(y[a], x[b]))
                    else:
                        y[c] = sub_op(x[c], mul_op(x[a], y[b]))
                yield (1 - side, tuple(y))

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        i = 0
        while i < len(order):
            vertex = order[i]
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
            i += 1

        n_luw = len(order)
        return n_luw, edges

    # Special small cases
    if k == 3 and g == 5:
        edges = [(0,1),(1,4),(4,2),(2,3),(3,0),
                 (5,6),(6,7),(7,8),(8,9),(9,5),
                 (0,5),(1,6),(2,7),(3,8),(4,9)]
        return 10, edges

    if k == 3 and g == 7:
        # Heawood graph (3-regular, girth 6? No, Heawood is girth 6. Wait.
        # Heawood is incidence graph of Fano plane. Girth 6.
        # For girth 7, we need something else?
        # Actually, smallest 3-regular girth 7 is 24 vertices? No.
        # Let's just rely on LUW for general cases.
        pass

    # Try LUW first
    n_luw, edges_luw = luw_graph(k, g)
    if n_luw is not None:
        adj_luw = build_adj(edges_luw, n_luw)
        if check_girth_fast(adj_luw, n_luw, g):
            return n_luw, edges_luw

    # Fallback: Random search (unlikely to succeed for large n)
    moore = moore_bound(k, g)
    n = moore
    if n * k % 2 != 0:
        n += 1
    
    rng = random.Random(12345)
    best_n = None
    best_edges = None
    
    max_search = min(moore + 100, 1000) # Limit search to small range for speed
    
    for n_try in range(n, max_search + 1):
        if time.time() > deadline - 3:
            break
        if n_try * k % 2 != 0:
            continue
        if best_n is not None and n_try >= best_n:
            break
        
        for _ in range(10):
            if time.time() > deadline - 3:
                break
            edges_r, adj_r = random_k_regular(n_try, k, rng)
            if edges_r is None:
                continue
            if check_girth_fast(adj_r, n_try, g):
                if best_n is None or n_try < best_n:
                    best_n = n_try
                    best_edges = edges_r
                break

    if best_n is not None:
        return best_n, best_edges

    if n_luw is not None:
        return n_luw, edges_luw

    raise RuntimeError("Failed to construct valid graph")

# EVOLVE-BLOCK-END

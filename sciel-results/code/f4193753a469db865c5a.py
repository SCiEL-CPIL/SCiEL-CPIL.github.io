import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.monotonic()
    deadline = start_time + 110  # safety margin before 120s kill

    def time_left(margin=5.0):
        return deadline - time.monotonic() - margin

    # ------------------------------------------------------------------ #
    # LUW construction CD(d, q) -> connected component of the zero point #
    # ------------------------------------------------------------------ #
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if d + 5 < g:
            d += 1
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: (x * y) % q
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

    # ------------------------------------------------------------------ #
    # Exact girth verification via non-backtracking walks from every vertex
    # ------------------------------------------------------------------ #
    def girth_ok(adj, n, g):
        for s in range(n):
            seen = set()
            dq = deque()
            for w in adj[s]:
                st = (w, s)
                seen.add(st)
                dq.append((st, 1))
            while dq:
                (v, p), d = dq.popleft()
                if d >= g:
                    continue
                for w in adj[v]:
                    if w == p:
                        continue
                    if w == s:
                        return False  # cycle of length d+1 < g
                    st = (w, v)
                    if st not in seen:
                        seen.add(st)
                        dq.append((st, d + 1))
        return True

    # ------------------------------------------------------------------ #
    # Cayley graph on Z_q x Z_2 with degree 5 (generators +/-a, +/-b, (0,1))
    # ------------------------------------------------------------------ #
    def cayley_zq_z2(q, a, b):
        n = 2 * q
        adj = [[] for _ in range(n)]
        edges = []
        for i in range(q):
            for t in (0, 1):
                u = 2 * i + t
                for (di, dt) in ((a, 0), (-a, 0), (b, 0), (-b, 0), (0, 1)):
                    v = 2 * ((i + di) % q) + ((t + dt) & 1)
                    if u < v:
                        adj[u].append(v)
                        adj[v].append(u)
                        edges.append((u, v))
        return n, adj, edges

    def is_prime(x):
        if x < 2:
            return False
        if x % 2 == 0:
            return x == 2
        d = 3
        while d * d <= x:
            if x % d == 0:
                return False
            d += 2
        return True

    def next_prime(x):
        x = max(2, x)
        while not is_prime(x):
            x += 1
        return x

    # ------------------------------------------------------------------ #
    # Random regular graph via configuration model
    # ------------------------------------------------------------------ #
    def random_regular_graph(n, k, rng, max_attempts=50):
        if n * k % 2 != 0:
            return None
        for _ in range(max_attempts):
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

    def try_edge_swap(adj, n, rng, attempts=100):
        for _ in range(attempts):
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

    # ------------------------------------------------------------------ #
    # Main: start with LUW baseline, then try to beat it
    # ------------------------------------------------------------------ #
    n_luw, edges_luw = luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw

    # Moore bound (odd girth g = 2r+1): n >= 1 + k*(1 + (k-1) + ... + (k-1)^(r-1))
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(r))

    # --- Phase 1: Scan Cayley graph family Z_q x Z_2 for smaller valid graphs ---
    lo_q = max(2, (moore_bound + 1) // 2)
    hi_q = best_n // 2

    # Generator pairs for the Cayley graph
    pairs = []
    for a in range(1, 12):
        for b in range(1, 12):
            if b != a and b != (-a) % 10**9:
                pairs.append((a, b))
    # Deduplicate
    seen_pairs = set()
    pairs_dedup = []
    for p in pairs:
        if p not in seen_pairs:
            seen_pairs.add(p)
            pairs_dedup.append(p)
    pairs = pairs_dedup

    q = lo_q
    while q <= hi_q and time_left(3.0) > 0:
        q = next_prime(q)
        for (a, b) in pairs:
            if q <= max(a, b):
                continue
            if time_left(2.0) <= 0:
                break
            a_mod = a % q
            b_mod = b % q
            if a_mod == 0 or b_mod == 0 or a_mod == b_mod or a_mod == (-b_mod) % q:
                continue
            n_cay, adj_cay, edges_cay = cayley_zq_z2(q, a_mod, b_mod)
            if n_cay < best_n and girth_ok(adj_cay, n_cay, g):
                best_n = n_cay
                best_edges = edges_cay
                # Update hi_q since we found a smaller one
                hi_q = best_n // 2
                if 2 * q >= moore_bound:
                    break
        if time_left(3.0) <= 0:
            break
        q += 1

    # --- Phase 2: Random regular graph search + edge swaps ---
    if time_left(10.0) > 0:
        rng = random.Random(42)
        search_lo = max(moore_bound, k + 1)
        search_hi = best_n

        for n in range(search_lo, search_hi):
            if time_left(8.0) < 0:
                break
            if n * k % 2 != 0:
                continue
            for attempt in range(20):
                if time_left(5.0) < 0:
                    break
                edges = random_regular_graph(n, k, rng, max_attempts=30)
                if edges is None:
                    continue
                adj = build_adj(n, edges)
                if girth_ok(adj, n, g):
                    best_n = n
                    best_edges = [(min(u, v), max(u, v)) for u, v in edges]
                    break
                # Try edge swaps to fix girth violations
                for _ in range(200):
                    if time_left(3.0) < 0:
                        break
                    if not try_edge_swap(adj, n, rng, attempts=50):
                        break
                    if girth_ok(adj, n, g):
                        best_n = n
                        best_edges = [(min(u, v), max(u, v)) for u in range(n) for v in adj[u] if u < v]
                        break
            if best_n < n:
                break

    return best_n, best_edges

import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    def remaining():
        return deadline - time.time()

    # ---------- arithmetic helpers ----------
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def is_prime_power(n):
        if n < 2:
            return False
        if is_prime(n):
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if n % p == 0:
                t = n
                while t % p == 0:
                    t //= p
                return t == 1
        return False

    def modinv(a, m):
        if m <= 1:
            return None
        try:
            return pow(a, -1, m)
        except (ValueError, ZeroDivisionError):
            return None

    def moore_bound(k, g):
        if k <= 1:
            return 0
        if g == 1:
            return 1
        r = g // 2
        if g % 2 == 1:  # g = 2r + 1
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:  # g = 2r
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)

    # ---------- graph validation ----------
    def check_girth(adj, n, g):
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = deque([u])
            while queue:
                x = queue.popleft()
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dist[x] + 1
                        parent[y] = x
                        queue.append(y)
                    elif y != parent[x] and parent[y] != x:
                        if dist[x] + dist[y] + 1 < g:
                            return False
        return True

    def is_connected(adj, n):
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        count = 0
        while queue:
            u = queue.popleft()
            count += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return count == n

    def validate(adj, n, k, g):
        for i in range(n):
            if len(adj[i]) != k:
                return False
        if not is_connected(adj, n):
            return False
        if not check_girth(adj, n, g):
            return False
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return (n, edges)

    # ---------- Lazebnik-Ustimenko-Woldar baseline (deterministic) ----------
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

    # ---------- PGL(2, q) machinery (q odd prime power) ----------
    def action_on_point(x, a, b, c, d, q):
        if x == q:  # infinity
            if c == 0:
                return q
            inv_c = modinv(c, q)
            if inv_c is None:
                return None
            return (a * inv_c) % q
        num = (a * x + b) % q
        den = (c * x + d) % q
        if den == 0:
            return q
        inv_den = modinv(den, q)
        if inv_den is None:
            return None
        return (num * inv_den) % q

    def inv_matrix(a, b, c, d, q):
        det = (a * d - b * c) % q
        det_inv = modinv(det, q)
        if det_inv is None:
            return None
        return (d * det_inv % q, -b * det_inv % q, -c * det_inv % q, a * det_inv % q)

    def is_derangement(a, b, c, d, q):
        # fixed points of x -> (a x + b)/(c x + d)
        # infinity fixed iff c == 0
        if c == 0:
            return False
        # finite fixed: c x^2 + (d - a) x - b = 0, discriminant (d-a)^2 + 4 b c
        disc = ((d - a) ** 2 + 4 * b * c) % q
        if disc == 0:
            return False  # exactly one finite fixed point
        # two finite fixed points iff disc is a non-zero square mod q
        return pow(disc, (q - 1) // 2, q) != 1

    def build_from_gens(gens, q, k, g):
        n = q + 1
        adj = [set() for _ in range(n)]
        for i in range(n):
            x = i if i < q else q
            for gen in gens:
                y = action_on_point(x, gen[0], gen[1], gen[2], gen[3], q)
                if y is None or y == i:
                    return None
                j = y if y < q else q
                adj[i].add(j)
                adj[j].add(i)
        return validate(adj, n, k, g)

    # deterministic structured generator set
    def try_structured_pgl(q, k, g):
        if q % 2 == 0 or not is_prime_power(q):
            return None
        n = q + 1
        if n < n_min:
            return None

        def is_nonsquare(a):
            a %= q
            if a == 0:
                return True
            return pow(a, (q - 1) // 2, q) != 1

        # G_c: x -> (x + c)/x = (1, c, 1, 0); derangement iff 1 + 4c is a non-square
        cands = []
        for cc in range(1, q):
            if is_nonsquare(1 + 4 * cc):
                cands.append(cc)
            if len(cands) >= 6:
                break
        if len(cands) < 2:
            return None

        # H: x -> (x + 1)/(x - 1) = (1, 1, 1, q-1); derangement iff 8 is a non-square
        h_ok = is_nonsquare(8)
        # G'_c: x -> (x - c)/(x + c) = (1, -c, 1, c); disc = 4 + 4(-c)(c) = 4(1 - c^2)
        # derangement iff 1 - c^2 is a non-square
        h2_cands = []
        for cc in range(1, q):
            if is_nonsquare(1 - cc * cc):
                h2_cands.append(cc)
            if len(h2_cands) >= 4:
                break

        c1, c2 = cands[0], cands[1]
        g1 = (1, c1, 1, 0)
        g1i = inv_matrix(*g1, q)
        g2 = (1, c2, 1, 0)
        g2i = inv_matrix(*g2, q)
        if g1i is None or g2i is None:
            return None
        h = (1, 1, 1, q - 1)

        if h_ok:
            res = build_from_gens([g1, g1i, g2, g2i, h], q, k, g)
            if res:
                return res

        # fallback: two G pairs + one G'_c
        if len(h2_cands) >= 1:
            cc = h2_cands[0]
            g3 = (1, q - cc, 1, cc)
            res = build_from_gens([g1, g1i, g2, g2i, g3], q, k, g)
            if res:
                return res
        return None

    # random search in PGL(2, q)
    def random_search_pgl(q, k, g, budget):
        n = q + 1
        if n < n_min or n >= best_n[0]:
            return None
        if q % 2 == 0 or not is_prime_power(q):
            return None

        def random_gen():
            while True:
                a = random.randrange(q)
                b = random.randrange(q)
                c = random.randrange(q)
                d = random.randrange(q)
                if (a * d - b * c) % q != 0:
                    return (a, b, c, d)

        def random_gens():
            out = []
            for _ in range(k):
                m = random_gen()
                if random.random() < 0.5:
                    mi = inv_matrix(*m, q)
                    if mi is not None:
                        out.extend([m, mi])
                        if len(out) >= k:
                            return out[:k]
                else:
                    out.append(m)
            return out

        seen = set()
        while time.time() < budget:
            gens = random_gens()
            key = tuple(gens)
            if key in seen:
                continue
            seen.add(key)
            if any((a * d - b * c) % q == 0 for (a, b, c, d) in gens):
                continue
            if not all(is_derangement(*gen, q) for gen in gens):
                continue
            res = build_from_gens(gens, q, k, g)
            if res:
                return res
        return None

    # ---------- main ----------
    # 1. deterministic baseline
    n_luw, edges_luw = luw_construct(k, g)
    best_n = [n_luw]
    best_result = (n_luw, edges_luw)

    # 2. deterministic structured PGL for small q
    q_max_struct = min(best_n[0] - 2, 60)
    for curr_q in range(max(3, n_min - 1), q_max_struct + 1):
        if time.time() > deadline:
            break
        res = try_structured_pgl(curr_q, k, g)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res
            return best_result  # smallest q first => smallest n in this family

    # 3. random PGL search, time-boxed, q increasing
    q_cursor = max(3, n_min - 1)
    while q_cursor + 1 < best_n[0] and time.time() < deadline:
        budget = min(deadline - 2.0, time.time() + 1.0)
        res = random_search_pgl(q_cursor, k, g, budget)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res
        q_cursor += 1

    return best_result

# EVOLVE-BLOCK-END

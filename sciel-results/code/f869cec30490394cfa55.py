import time
import math
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # safety margin below 150s wall-clock

    def elapsed():
        return time.time() - start_time

    def moore_bound(k, g):
        if k <= 1:
            return 0
        if g == 1:
            return 1
        r = g // 2
        if g % 2 == 1:
            # g = 2r+1
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:
            # g = 2r
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)

    def girth_of(adj, n):
        best = float('inf')
        for u in range(n):
            dist = {}
            q = deque()
            for v in adj[u]:
                dist[(v, u)] = 1
                q.append((v, u))
            while q:
                v, parent = q.popleft()
                d = dist[(v, parent)]
                for w in adj[v]:
                    if w == parent:
                        continue
                    if w == u:
                        if d + 1 < best:
                            best = d + 1
                        continue
                    state = (w, v)
                    if state not in dist:
                        dist[state] = d + 1
                        q.append(state)
        return best

    def is_valid(k, g, n, edges):
        adj = [set() for _ in range(n)]
        seen = set()
        for (i, j) in edges:
            if i == j:
                return False
            if i < 0 or i >= n or j < 0 or j >= n:
                return False
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                return False
            seen.add((a, b))
            adj[i].add(j)
            adj[j].add(i)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        visited = [False] * n
        dq = deque([0])
        visited[0] = True
        cnt = 0
        while dq:
            u = dq.popleft()
            cnt += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    dq.append(v)
        if cnt != n:
            return False
        gr = girth_of(adj, n)
        return gr >= g

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

    def primitive_root(p):
        if p == 2:
            return 1
        n = p - 1
        factors = []
        m = n
        d = 2
        while d * d <= m:
            if m % d == 0:
                factors.append(d)
                while m % d == 0:
                    m //= d
            d += 1
        if m > 1:
            factors.append(m)
        for g in range(2, p):
            ok = True
            for f in factors:
                if pow(g, n // f, p) == 1:
                    ok = False
                    break
            if ok:
                return g
        return None

    def build_morgenstern(q, k, g, rng):
        n = q + 1
        def action(x, a, b, c, d):
            if x == q:
                if c == 0:
                    return q
                inv_c = pow(c, -1, q)
                return (a * inv_c) % q
            num = (a * x + b) % q
            den = (c * x + d) % q
            if den == 0:
                return q
            return (num * pow(den, -1, q)) % q

        def try_gens(gens):
            adj = [set() for _ in range(n)]
            for i in range(n):
                x = i
                for (a, b, c, d) in gens:
                    y = action(x, a, b, c, d)
                    if i == y:
                        return None
                    if y in adj[i]:
                        return None
                    adj[i].add(y)
                    adj[y].add(i)
            for i in range(n):
                if len(adj[i]) != k:
                    return None
            visited = [False] * n
            dq = deque([0])
            visited[0] = True
            cnt = 0
            while dq:
                u = dq.popleft()
                cnt += 1
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        dq.append(v)
            if cnt != n:
                return None
            gr = girth_of(adj, n)
            if gr < g:
                return None
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges

        # Local search for generators
        def random_gen():
            while True:
                a = rng.randint(0, q - 1)
                b = rng.randint(0, q - 1)
                c = rng.randint(0, q - 1)
                d = rng.randint(0, q - 1)
                if (a * d - b * c) % q != 0:
                    return (a, b, c, d)

        # Initial set: random
        gens = [random_gen() for _ in range(k)]
        
        best_gens = None
        best_girth = 0
        
        for _ in range(500): # 500 iterations of local search
            if elapsed() > deadline - 5:
                break
            res = try_gens(gens)
            if res:
                return res
            
            # If invalid, try to improve
            # Find shortest cycle in the graph built by gens (even if invalid)
            # We need to build the graph to find the cycle
            adj = [set() for _ in range(n)]
            valid_structure = True
            for i in range(n):
                x = i
                for idx, (a, b, c, d) in enumerate(gens):
                    y = action(x, a, b, c, d)
                    if i == y:
                        valid_structure = False
                        break
                    if y in adj[i]:
                        valid_structure = False
                        break
                    adj[i].add(y)
                    adj[y].add(i)
                if not valid_structure:
                    break
            
            if not valid_structure:
                # Replace a random generator
                idx = rng.randint(0, k-1)
                gens[idx] = random_gen()
                continue
            
            # Find shortest cycle
            gr = girth_of(adj, n)
            if gr > best_girth:
                best_girth = gr
                best_gens = gens[:]
            
            if gr >= g:
                return try_gens(gens)
            
            # Identify a generator in the shortest cycle
            # This is complex. Simpler: pick a random generator and swap.
            idx = rng.randint(0, k-1)
            gens[idx] = random_gen()
            
        if best_gens and best_girth >= g:
            return try_gens(best_gens)
            
        return None

    def luw_construct(k, g, max_d):
        q = k
        def luw_for_d(d):
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

        # Try increasing d until valid
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        while d <= max_d:
            if elapsed() > deadline - 10:
                break
            n, edges = luw_for_d(d)
            if is_valid(k, g, n, edges):
                return n, edges
            d += 1
        return None

    best = None
    best_n = float('inf')

    def consider(n, edges):
        nonlocal best, best_n
        if best is None or n < best_n:
            if is_valid(k, g, n, edges):
                best = (n, edges)
                best_n = n
                return True
        return False

    rng = random.Random(12345)

    # 1) Try Morgenstern
    max_q = max(n_min + 500, 1000)
    q = max(2, n_min - 1)
    while q <= max_q and elapsed() < deadline - 30:
        if is_prime(q):
            res = build_morgenstern(q, k, g, rng)
            if res:
                consider(res[0], res[1])
        q += 1

    # 2) Try LUW with increasing d
    if elapsed() < deadline - 5:
        res = luw_construct(k, g, max_d=10)
        if res:
            consider(res[0], res[1])

    if best is None:
        # Last resort: LUW with very large d
        res = luw_construct(k, g, max_d=20)
        if res:
            best = res
        else:
            # Should not happen
            best = (0, [])

    return best

# EVOLVE-BLOCK-END

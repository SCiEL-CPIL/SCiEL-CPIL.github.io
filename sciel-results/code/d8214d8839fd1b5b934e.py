import time
import math
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before the 150 s kill

    # ---------- LUW construction (guaranteed fallback) ----------
    def luw_construct(k, g, d_override=None, rule_override=None):
        q = k
        if d_override is not None:
            d = d_override
        else:
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
        if rule_override is not None:
            rule = rule_override
        else:
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

    def gen_rules(d, variant=0):
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            if variant == 0:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            elif variant == 1:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 1), (c0, 1), (c0 + 1, 1), (1, c0)][r]
            elif variant == 2:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(2, c0 - 1), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            elif variant == 3:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][(r + 1) % 4]
            elif variant == 4:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][(r + 2) % 4]
            elif variant == 5:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][(r + 3) % 4]
            elif variant == 6:
                rule[c] = (1, c - 2) if c % 2 == 1 else (c - 1, 1)
            elif variant == 7:
                rule[c] = (1, c - 2) if c % 3 == 0 else ((c - 1, 1) if c % 3 == 1 else (c, 1))
            else:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return rule

    # ---------- girth verification (non-backtracking BFS, task contract) ----------
    def verify_girth(n, edges, target_girth):
        adj = [[] for _ in range(n)]
        seen = set()
        for u, v in edges:
            if u == v:
                return False
            if u < 0 or v < 0 or u >= n or v >= n:
                return False
            key = (u, v) if u < v else (v, u)
            if key in seen:
                return False
            seen.add(key)
            adj[u].append(v)
            adj[v].append(u)
        for u in range(n):
            if len(adj[u]) != k:
                return False
        half = (target_girth - 1) // 2
        for s in range(n):
            dist = {}
            queue = deque()
            for nb in adj[s]:
                st = (nb, s)
                dist[st] = 1
                queue.append(st)
            while queue:
                v, p = queue.popleft()
                dcur = dist[(v, p)]
                if dcur >= half:
                    continue
                for w in adj[v]:
                    if w == p:
                        continue
                    st = (w, v)
                    if st in dist:
                        if dcur + dist[st] + 1 < target_girth:
                            return False
                    else:
                        dist[st] = dcur + 1
                        queue.append(st)
        return True

    if k != 3 or g != 16:
        return luw_construct(k, g)

    # ---------- Petersen-graph voltage lift with finite-field (algebraic) voltages ----------
    def petersen_voltage_lift(k, g, deadline):
        # Petersen graph: 3-regular, girth 5, 10 vertices, 15 edges.
        pe = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
              (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
              (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
        n_base = 10
        nb_adj = [[] for _ in range(n_base)]
        for ei, (u, v) in enumerate(pe):
            nb_adj[u].append((v, ei))
            nb_adj[v].append((u, ei))

        # Precompute all non-backtracking closed walks of length 3..g-1
        # We only need the edge count signatures for abelian groups
        signatures = set()
        for sv in range(n_base):
            stack = []
            for nv, ei in nb_adj[sv]:
                stack.append((nv, ei, 1, [0]*15, [ei]))
            while stack:
                cv, pei, L, counts, path = stack.pop()
                if L >= g:
                    continue
                if cv == sv and L >= 3:
                    # Only add if L is odd? No, girth is even 16. 
                    # But we must check all L < 16.
                    signatures.add(tuple(counts))
                for nv, ei in nb_adj[cv]:
                    if ei == pei:
                        continue
                    new_counts = counts[:]
                    new_counts[ei] += 1
                    stack.append((nv, ei, L + 1, new_counts, path + [ei]))
        
        # Convert to list for faster iteration
        sig_list = list(signatures)
        W = len(sig_list)
        
        def is_prime(x):
            if x < 2:
                return False
            if x == 2:
                return True
            if x % 2 == 0:
                return False
            i = 3
            while i * i <= x:
                if x % i == 0:
                    return False
                i += 2
            return True

        # Candidate moduli: primes with 10*m >= 512 (Moore bound)
        # We want small m to minimize n = 10*m
        primes = []
        x = 53
        while len(primes) < 20:
            if is_prime(x):
                primes.append(x)
            x += 1

        rng = random.Random(12345)

        def build_lift(m, volts):
            return [(u * m + t, v * m + (t + vol) % m)
                    for (u, v), vol in zip(pe, volts)
                    for t in range(m)]

        def precheck(m, volts):
            for sig in sig_list:
                s = 0
                for i in range(15):
                    if sig[i]:
                        s += sig[i] * volts[i]
                if s % m == 0:
                    return False
            return True

        best = None
        for m in primes:
            if time.time() > deadline - 8.0:
                break
            if 10 * m >= (best[0] if best else 10 ** 9):
                continue
            
            # Try many random assignments
            # We can also try structured ones
            found = False
            trials = 0
            max_trials = 5000
            while trials < max_trials and time.time() < deadline - 6.0:
                volts = [rng.randrange(1, m) for _ in range(15)]
                if precheck(m, volts):
                    edges = build_lift(m, volts)
                    # Final verification
                    if verify_girth(10 * m, edges, g):
                        return 10 * m, edges
                trials += 1
            
            # If random fails, try some structured ones?
            # For now, move to next prime
            
        return None

    # ---------- main search ----------
    best_n = float('inf')
    best_edges = None

    # 1) algebraic Petersen voltage lift (aim well below 936)
    res = petersen_voltage_lift(k, g, deadline)
    if res is not None:
        best_n, best_edges = res

    # 2) LUW family (guaranteed valid, fallback)
    # Only use if voltage lift failed or if we have plenty of time
    if time.time() < deadline - 30.0:
        for d in [7, 8]:
            if time.time() > deadline - 5.0:
                break
            max_variants = 5
            for variant in range(max_variants):
                if time.time() > deadline - 5.0:
                    break
                rule = gen_rules(d, variant)
                n, edges = luw_construct(k, g, d_override=d, rule_override=rule)
                if n >= best_n:
                    continue
                if verify_girth(n, edges, g):
                    best_n = n
                    best_edges = edges

    if best_edges is not None:
        return best_n, best_edges

    return luw_construct(k, g)

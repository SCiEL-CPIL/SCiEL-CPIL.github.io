import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    if k != 3:
        return _construct_luw(k, g, d=7)

    best_n = float('inf')
    best_edges = None

    # Primary: Voltage lift of Petersen graph with constraint-based search
    try:
        n, edges = _voltage_petersen_optimized(k, g, deadline)
        if edges is not None and n < best_n:
            if _has_girth_at_least(n, edges, g):
                best_n = n
                best_edges = edges
    except Exception:
        pass

    # Secondary: Voltage lift of K4 (complete graph on 4 vertices)
    if best_edges is None or best_n > 936:
        try:
            n, edges = _voltage_k4_optimized(k, g, deadline)
            if edges is not None and n < best_n:
                if _has_girth_at_least(n, edges, g):
                    best_n = n
                    best_edges = edges
        except Exception:
            pass

    # Tertiary: Cayley graph on C_m ⋊ C_3
    if best_edges is None or best_n > 936:
        try:
            n, edges = _construct_cayley_c3(k, g, deadline)
            if edges is not None and n < best_n:
                if _has_girth_at_least(n, edges, g):
                    best_n = n
                    best_edges = edges
        except Exception:
            pass

    # Fallback: LUW construction
    if best_edges is None:
        try:
            for d_try in [7, 8, 9, 10, 11]:
                if time.time() > deadline - 10:
                    break
                n, edges = _construct_luw(k, g, d=d_try, rule_variant=0)
                if _has_girth_at_least(n, edges, g):
                    if n < best_n:
                        best_n = n
                        best_edges = edges
        except Exception:
            pass

    if best_edges is not None:
        return best_n, best_edges

    # Final fallback
    return _construct_luw(k, g, d=7)

def _voltage_petersen_optimized(k, g, deadline):
    """
    Voltage lift of Petersen graph with optimized constraint-based voltage search.
    """
    if k != 3:
        return None, None

    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]

    n_base = 10
    n_edges = 15

    adj = [[] for _ in range(n_base)]
    edge_index = {}
    for i, (u, v) in enumerate(petersen_edges):
        adj[u].append(v)
        adj[v].append(u)
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    # Generate all non-backtracking closed walks of length [5, g-1]
    max_walk_len = g - 1
    walk_vectors = set()

    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0] * n_edges)]
        while stack:
            v, prev_e, l, counts = stack.pop()
            if l >= max_walk_len:
                continue
            for next_v in adj[v]:
                if next_v == prev_e:
                    continue
                e_idx = edge_index[(v, next_v)]
                u, v_base = petersen_edges[e_idx]
                d = 1 if v == u else -1
                new_counts = counts.copy()
                new_counts[e_idx] += d
                new_l = l + 1
                if next_v == start_v and new_l >= 5:
                    walk_vectors.add(tuple(new_counts))
                stack.append((next_v, v, new_l, new_counts))

    # Filter and normalize constraints
    unique_constraints = set()
    for c in walk_vectors:
        if all(x == 0 for x in c):
            continue
        first_non_zero = -1
        for i in range(n_edges):
            if c[i] != 0:
                first_non_zero = i
                break
        if first_non_zero != -1:
            if c[first_non_zero] < 0:
                c = tuple(-x for x in c)
            unique_constraints.add(c)

    if not unique_constraints:
        return None, None

    C = np.array(list(unique_constraints), dtype=np.int64)

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

    best_n = float('inf')
    best_edges = None

    # Try primes in increasing order
    for m in range(53, 200):
        if time.time() > deadline - 10:
            break
        if not is_prime(m):
            continue
        if 10 * m >= best_n:
            break

        # Multiple restarts with local search
        np.random.seed(42 + m)
        random.seed(42 + m)

        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break

            v = np.random.randint(1, m, size=n_edges).astype(np.int64)

            def cost(v):
                residues = np.mod(C @ v, m)
                return np.sum(residues == 0)

            best_v = v.copy()
            best_cost = cost(v)

            if best_cost == 0:
                n = 10 * m
                edges = []
                for i, (u, v_base) in enumerate(petersen_edges):
                    vol = v[i]
                    for layer in range(m):
                        u_lift = u * m + layer
                        v_lift = v_base * m + (layer + vol) % m
                        if u_lift < v_lift:
                            edges.append((u_lift, v_lift))
                        else:
                            edges.append((v_lift, u_lift))
                edges = list(set(edges))
                if _has_girth_at_least(n, edges, g):
                    if n < best_n:
                        best_n = n
                        best_edges = edges
                        if n <= 936:
                            return n, edges
                continue

            # Local search: hill climbing with random restarts
            T = 1.0
            cooling = 0.9998
            min_T = 0.01
            max_iters = 2000

            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break

                j = random.randint(0, n_edges - 1)
                old_val = v[j]
                v[j] = random.randint(1, m - 1)

                new_cost = cost(v)
                delta = new_cost - best_cost

                if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_v = v.copy()
                    if new_cost == 0:
                        break
                else:
                    v[j] = old_val

                T *= cooling

            if best_cost == 0:
                n = 10 * m
                edges = []
                for i, (u, v_base) in enumerate(petersen_edges):
                    vol = best_v[i]
                    for layer in range(m):
                        u_lift = u * m + layer
                        v_lift = v_base * m + (layer + vol) % m
                        if u_lift < v_lift:
                            edges.append((u_lift, v_lift))
                        else:
                            edges.append((v_lift, u_lift))
                edges = list(set(edges))
                if _has_girth_at_least(n, edges, g):
                    if n < best_n:
                        best_n = n
                        best_edges = edges
                        if n <= 936:
                            return n, edges

    if best_edges is not None:
        return best_n, best_edges
    return None, None

def _voltage_k4_optimized(k, g, deadline):
    """
    Voltage lift of K4 with optimized constraint-based voltage search.
    K4 has 6 edges, so voltage space is smaller.
    """
    if k != 3:
        return None, None

    k4_edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    n_base = 4
    n_edges = 6

    adj = [[] for _ in range(n_base)]
    edge_index = {}
    for i, (u, v) in enumerate(k4_edges):
        adj[u].append(v)
        adj[v].append(u)
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    # Generate non-backtracking closed walks of length [3, g-1]
    max_walk_len = g - 1
    walk_vectors = set()

    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0] * n_edges)]
        while stack:
            v, prev_e, l, counts = stack.pop()
            if l >= max_walk_len:
                continue
            for next_v in adj[v]:
                if next_v == prev_e:
                    continue
                e_idx = edge_index[(v, next_v)]
                u, v_base = k4_edges[e_idx]
                d = 1 if v == u else -1
                new_counts = counts.copy()
                new_counts[e_idx] += d
                new_l = l + 1
                if next_v == start_v and new_l >= 3:
                    walk_vectors.add(tuple(new_counts))
                stack.append((next_v, v, new_l, new_counts))

    unique_constraints = set()
    for c in walk_vectors:
        if all(x == 0 for x in c):
            continue
        first_non_zero = -1
        for i in range(n_edges):
            if c[i] != 0:
                first_non_zero = i
                break
        if first_non_zero != -1:
            if c[first_non_zero] < 0:
                c = tuple(-x for x in c)
            unique_constraints.add(c)

    if not unique_constraints:
        return None, None

    C = np.array(list(unique_constraints), dtype=np.int64)

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

    best_n = float('inf')
    best_edges = None

    for m in range(100, 300):
        if time.time() > deadline - 10:
            break
        if not is_prime(m):
            continue
        if 4 * m >= best_n:
            break

        np.random.seed(123 + m)
        random.seed(123 + m)

        max_restarts = 100
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break

            v = np.random.randint(1, m, size=n_edges).astype(np.int64)

            def cost(v):
                residues = np.mod(C @ v, m)
                return np.sum(residues == 0)

            best_v = v.copy()
            best_cost = cost(v)

            if best_cost == 0:
                n = 4 * m
                edges = []
                for i, (u, v_base) in enumerate(k4_edges):
                    vol = v[i]
                    for layer in range(m):
                        u_lift = u * m + layer
                        v_lift = v_base * m + (layer + vol) % m
                        if u_lift < v_lift:
                            edges.append((u_lift, v_lift))
                        else:
                            edges.append((v_lift, u_lift))
                edges = list(set(edges))
                if _has_girth_at_least(n, edges, g):
                    if n < best_n:
                        best_n = n
                        best_edges = edges
                        if n <= 936:
                            return n, edges
                continue

            T = 1.0
            cooling = 0.9997
            min_T = 0.01
            max_iters = 3000

            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break

                j = random.randint(0, n_edges - 1)
                old_val = v[j]
                v[j] = random.randint(1, m - 1)

                new_cost = cost(v)
                delta = new_cost - best_cost

                if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_v = v.copy()
                    if new_cost == 0:
                        break
                else:
                    v[j] = old_val

                T *= cooling

            if best_cost == 0:
                n = 4 * m
                edges = []
                for i, (u, v_base) in enumerate(k4_edges):
                    vol = best_v[i]
                    for layer in range(m):
                        u_lift = u * m + layer
                        v_lift = v_base * m + (layer + vol) % m
                        if u_lift < v_lift:
                            edges.append((u_lift, v_lift))
                        else:
                            edges.append((v_lift, u_lift))
                edges = list(set(edges))
                if _has_girth_at_least(n, edges, g):
                    if n < best_n:
                        best_n = n
                        best_edges = edges
                        if n <= 936:
                            return n, edges

    if best_edges is not None:
        return best_n, best_edges
    return None, None

def _construct_cayley_c3(k, g, deadline):
    """
    Construct a 3-regular Cayley graph on the non-abelian group G = C_m ⋊ C_3.
    """
    if k != 3:
        return None, None

    best_n = float('inf')
    best_edges = None

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

    def primitive_root_prime(p):
        if p == 2:
            return 1
        phi = p - 1
        factors = []
        d = 2
        while d * d <= phi:
            if phi % d == 0:
                factors.append(d)
                while phi % d == 0:
                    phi //= d
            d += 1
        if phi > 1:
            factors.append(phi)
        for g in range(2, p):
            valid = True
            for f in factors:
                if pow(g, (p - 1) // f, p) == 1:
                    valid = False
                    break
            if valid:
                return g
        return None

    candidates = []
    m = 7
    while m < 2000:
        if m % 3 == 1 and is_prime(m):
            candidates.append(m)
        m += 1

    candidates.sort()

    for m in candidates:
        if time.time() > deadline - 10:
            break
        n = 3 * m
        if n >= best_n:
            break

        g_prim = primitive_root_prime(m)
        if g_prim is None:
            continue

        omega = pow(g_prim, (m - 1) // 3, m)

        if pow(omega, 3, m) != 1 or omega == 1:
            continue

        edges = []
        for a in range(m):
            for b in range(3):
                v_idx = a * 3 + b
                a_s = (a + pow(omega, b, m)) % m
                v_s = a_s * 3 + b
                if v_idx < v_s:
                    edges.append((v_idx, v_s))
                a_si = (a - pow(omega, b, m)) % m
                v_si = a_si * 3 + b
                if v_idx < v_si:
                    edges.append((v_idx, v_si))
                b_t = (b + 1) % 3
                v_t = a * 3 + b_t
                if v_idx < v_t:
                    edges.append((v_idx, v_t))

        edges = list(set(edges))

        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != 3 for d in deg):
            continue

        if _has_girth_at_least(n, edges, g):
            if n < best_n:
                best_n = n
                best_edges = edges
                if n <= 936:
                    return n, edges

    if best_edges is not None:
        return best_n, best_edges
    return None, None

def _construct_luw(k, g, d=None, rule_variant=0):
    """
    LUW construction with specified depth d and rule variant.
    """
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)

    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    if rule_variant == 0:
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    elif rule_variant == 1:
        rule = {2: (1, 1), 3: (1, 2), 4: (2, 1)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 1), (c0 - 2, 1), (c0, 1), (1, c0 + 2)][r]
    elif rule_variant == 2:
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 3
            c0 = c - (c - 5) % 3
            rule[c] = [(1, c0 - 1), (c0, 1), (1, c0 + 1)][r]
    else:
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 5
            c0 = c - (c - 5) % 5
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1), (1, c0 + 2)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges

def _has_girth_at_least(n, edges, g):
    """
    Check if graph has girth at least g using non-backtracking walks.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for start in range(n):
        parent = [-1] * n
        level = [-1] * n
        level[start] = 0
        queue = [start]
        head = 0

        while head < len(queue):
            u = queue[head]
            head += 1

            if level[u] > (g - 1) // 2:
                continue

            for v in adj[u]:
                if v == parent[u]:
                    continue
                if level[v] == -1:
                    level[v] = level[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = level[u] + level[v] + 1
                    if cycle_len < g:
                        return False

    return True

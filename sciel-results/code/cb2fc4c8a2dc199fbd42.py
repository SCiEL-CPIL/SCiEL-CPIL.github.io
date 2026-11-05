import random
import time
import math
from collections import deque
import numpy as np


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    def remaining():
        return deadline - time.time()

    # LUW construction (deterministic, gives 576 for k=3,g=16)
    def _luw_graph(k, g):
        q = k
        d = max(3, (g - 2) // 2)
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

    # Fast girth verification using non-backtracking BFS
    def _verify_girth(n, edges, girth_req):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        for start in range(n):
            visited = set()
            queue = deque()
            for nb in adj[start]:
                queue.append((nb, start, 1))
                visited.add((nb, start))
            while queue:
                curr, prev, dist = queue.popleft()
                if dist >= girth_req:
                    continue
                if curr == start and dist >= 3:
                    return False
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    if (nb, curr) not in visited:
                        visited.add((nb, curr))
                        queue.append((nb, curr, dist + 1))
        return True

    # Voltage lift on Petersen (for smaller n attempts)
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    n_edges_base = len(base_edges)
    edge_index = {}
    for i, (u, v) in enumerate(base_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    base_adj = [[] for _ in range(n_base)]
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)

    def _get_walk_constraints(max_len):
        constraints = set()
        for start_v in range(n_base):
            stack = []
            for next_v in base_adj[start_v]:
                e_id = edge_index[(start_v, next_v)]
                u, v = base_edges[e_id]
                sign = 1 if u == start_v else -1
                coeff = [0] * n_edges_base
                coeff[e_id] = sign
                stack.append((next_v, start_v, 1, coeff))
            while stack:
                curr_v, prev_v, length, coeff = stack.pop()
                if length >= max_len:
                    continue
                if curr_v == start_v and length >= 3:
                    c = tuple(coeff)
                    for i in range(n_edges_base):
                        if c[i] != 0:
                            if c[i] < 0:
                                c = tuple(-x for x in c)
                            break
                    constraints.add(c)
                for next_v in base_adj[curr_v]:
                    if next_v == prev_v:
                        continue
                    e_id = edge_index[(curr_v, next_v)]
                    u, v = base_edges[e_id]
                    sign = 1 if u == curr_v else -1
                    new_coeff = coeff[:]
                    new_coeff[e_id] += sign
                    stack.append((next_v, curr_v, length + 1, new_coeff))
        return list(constraints)

    closed_walk_constraints = _get_walk_constraints(g)
    sparse_constraints = []
    for c in closed_walk_constraints:
        sparse = [(i, c[i]) for i in range(n_edges_base) if c[i] != 0]
        if sparse:
            sparse_constraints.append(sparse)
    seen = set()
    unique_constraints = []
    for sc in sparse_constraints:
        key = tuple(sorted(sc))
        if key not in seen:
            seen.add(key)
            unique_constraints.append(sc)
    sparse_constraints = unique_constraints

    C_dense = np.zeros((len(sparse_constraints), n_edges_base), dtype=np.int64)
    for i, sc in enumerate(sparse_constraints):
        for eid, c in sc:
            C_dense[i, eid] = c

    def check_voltage_valid(volts, m):
        v = np.array(volts, dtype=np.int64)
        residues = np.mod(C_dense @ v, m)
        return np.all(residues != 0)

    def construct_lift(volts, m):
        n = n_base * m
        edges = []
        for i, (u, v) in enumerate(base_edges):
            vol = volts[i]
            for t in range(m):
                u_lift = u * m + t
                v_lift = v * m + (t + vol) % m
                edges.append((u_lift, v_lift))
        return n, edges

    # Get LUW result as baseline
    luw_n, luw_edges = _luw_graph(k, g)
    best_n = luw_n
    best_edges = luw_edges

    random.seed(42)
    np.random.seed(42)

    # Try voltage lifts for smaller n (10*m vertices)
    # Moore bound for k=3, g=16: n >= 512, so m >= 52
    m_candidates = list(range(52, 70)) + list(range(70, 100, 2))

    for m in m_candidates:
        if remaining() < 30.0:
            break
        candidate_n = n_base * m
        if candidate_n >= best_n:
            continue

        # Try structured voltage assignments
        found = False
        for a in range(1, min(m, 15)):
            for b in range(0, min(m, 5)):
                if remaining() < 25.0:
                    break
                volts = [(a * (i + 1) + b) % m for i in range(n_edges_base)]
                g_val = m
                for v in volts:
                    g_val = math.gcd(g_val, v)
                    if g_val == 1:
                        break
                if g_val != 1:
                    continue
                if check_voltage_valid(volts, m):
                    n, edges = construct_lift(volts, m)
                    if _verify_girth(n, edges, g):
                        best_n = n
                        best_edges = edges
                        found = True
                        break
            if found:
                break
        if found:
            break

        # Random voltage search
        max_tries = 30000 if m < 60 else 15000
        for _ in range(max_tries):
            if remaining() < 25.0:
                break
            volts = [random.randint(1, m - 1) for _ in range(n_edges_base)]
            g_val = m
            for v in volts:
                g_val = math.gcd(g_val, v)
                if g_val == 1:
                    break
            if g_val != 1:
                continue
            if check_voltage_valid(volts, m):
                n, edges = construct_lift(volts, m)
                if _verify_girth(n, edges, g):
                    best_n = n
                    best_edges = edges
                    found = True
                    break
        if found:
            break

    return best_n, best_edges

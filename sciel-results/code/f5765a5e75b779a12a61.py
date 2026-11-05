import time
import random
import math
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    # ---------------- LUW Fallback Construction ----------------
    def _luw_graph(k, g):
        q = k
        d = max(3, (g - 2) // 2)
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
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

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
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # ---------------- Girth Verification ----------------
    def check_girth(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                d = dist[node]
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < g:
                            return cycle_len
                    else:
                        dist[nb] = d + 1
                        parent[nb] = node
                        if d + 1 < g:
                            queue.append(nb)
        return g

    # ---------------- Voltage Lift of Petersen ----------------
    def voltage_lift_petersen():
        if k != 3 or g != 16:
            return None

        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]

        n_base = 10
        edge_index = {}
        for i, (u, v) in enumerate(pet_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i

        adj = [[] for _ in range(n_base)]
        for u, v in pet_edges:
            adj[u].append(v)
            adj[v].append(u)

        # Generate all non-backtracking closed walks of length [5, g-1]
        max_L = g - 1
        walk_vectors = set()

        for start_v in range(n_base):
            stack = [(start_v, -1, 0, [0] * 15)]
            while stack:
                current, prev, length, counts = stack.pop()
                if length > max_L:
                    continue
                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = pet_edges[e_idx]
                    d = 1 if current == u else -1
                    new_counts = counts.copy()
                    new_counts[e_idx] += d
                    new_l = length + 1
                    if next_v == start_v and new_l >= 5:
                        c = tuple(new_counts)
                        if any(x != 0 for x in c):
                            first_nz = -1
                            for i in range(15):
                                if c[i] != 0:
                                    first_nz = i
                                    break
                            if first_nz != -1:
                                if c[first_nz] < 0:
                                    c = tuple(-x for x in c)
                            walk_vectors.add(c)
                    if new_l < max_L:
                        stack.append((next_v, current, new_l, new_counts))

        if not walk_vectors:
            return None

        C_list = list(walk_vectors)
        C = np.array(C_list, dtype=np.int64)
        n_constraints = len(C)
        n_vars = 15

        col_rows = []
        for j in range(n_vars):
            rows = np.where(C[:, j] != 0)[0]
            col_rows.append(rows)
        C_cols = [C[:, j] for j in range(n_vars)]

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

        def solve_voltage(m, max_iters=10000):
            v = np.random.randint(1, m, size=n_vars).astype(np.int64)
            residues = np.mod(C @ v, m)
            violated = (residues == 0).astype(np.float64)
            n_violated = int(violated.sum())

            lam = np.ones(n_constraints, dtype=np.float64)
            lam_scale = 0.1

            best_v = v.copy()
            best_n_violated = n_violated

            if best_n_violated == 0:
                return v, 0

            T = 2.0
            cooling = 0.9997
            min_T = 0.01
            stall_count = 0
            max_stall = 800
            mult_update_interval = 100

            for it in range(max_iters):
                if remaining() < 5:
                    break
                j = random.randint(0, n_vars - 1)
                old_val = v[j]
                new_val = random.randint(1, m - 1)
                if new_val == old_val:
                    continue
                v[j] = new_val
                delta_val = (new_val - old_val) % m
                col_j = C_cols[j]
                rows_j = col_rows[j]
                if len(rows_j) > 0:
                    old_res = residues[rows_j].copy()
                    residues[rows_j] = (old_res + col_j[rows_j] * delta_val) % m
                    new_violated = (residues[rows_j] == 0).astype(np.float64)
                    old_violated = (old_res == 0).astype(np.float64)
                    delta_lag = float(np.sum(lam[rows_j] * (new_violated - old_violated)))
                    if delta_lag < 0 or (T > min_T and random.random() < np.exp(-delta_lag / T)):
                        n_violated += int(new_violated.sum() - old_violated.sum())
                        if n_violated < best_n_violated:
                            best_n_violated = n_violated
                            best_v = v.copy()
                            stall_count = 0
                        if n_violated == 0:
                            return v, 0
                    else:
                        v[j] = old_val
                        residues[rows_j] = old_res
                        stall_count += 1
                T *= cooling
                if it % mult_update_interval == 0 and it > 0:
                    viol_count = (residues == 0).astype(np.float64)
                    lam = lam + lam_scale * viol_count
                    lam = np.clip(lam, 0.1, 100.0)
                if stall_count > max_stall:
                    v = np.random.randint(1, m, size=n_vars).astype(np.int64)
                    residues = np.mod(C @ v, m)
                    n_violated = int((residues == 0).sum())
                    if n_violated < best_n_violated:
                        best_n_violated = n_violated
                        best_v = v.copy()
                    T = 1.5
                    stall_count = 0
            return best_v, best_n_violated

        # Try primes in increasing order starting from Moore bound implication
        # Moore bound ~ 512, so 10p >= 512 => p >= 52
        m = 53
        while m < 200 and remaining() > 25:
            if is_prime(m):
                time_left = remaining()
                num_restarts = max(5, min(25, int(time_left / 5)))
                for restart in range(num_restarts):
                    if remaining() < 12:
                        break
                    v, cost = solve_voltage(m, max_iters=8000)
                    if cost == 0:
                        n = 10 * m
                        edges = []
                        for u, v_base in pet_edges:
                            e_idx = edge_index[(u, v_base)]
                            vol = v[e_idx]
                            for layer in range(m):
                                u_lift = u * m + layer
                                v_lift = v_base * m + (layer + vol) % m
                                if u_lift < v_lift:
                                    edges.append((u_lift, v_lift))
                                else:
                                    edges.append((v_lift, u_lift))
                        gc = check_girth(n, edges, g, deadline - 5)
                        if gc >= g:
                            return (n, edges)
            m += 1
        return None

    # Try voltage lift first
    res = voltage_lift_petersen()
    if res is not None:
        return res

    # Fallback: LUW construction
    n_luw, edges_luw = _luw_graph(k, g)
    return (n_luw, edges_luw)


if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")

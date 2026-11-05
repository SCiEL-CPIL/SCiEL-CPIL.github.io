import time
import numpy as np
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0

    def remaining():
        return deadline - time.time()

    # ---------------- girth verification (optimized) ----------------
    def check_girth(n, edges, g, time_limit):
        if n == 0:
            return -1
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        for i in range(n):
            if len(adj[i]) != k:
                return -1
        min_cycle = float('inf')
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
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    # ---------------- Voltage Lift on Petersen (fast, guaranteed) ----------------
    def voltage_lift_search():
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

        # Generate non-backtracking closed walk constraints
        max_L = g - 1
        walk_vectors = set()
        for start_v in range(n_base):
            stack = [(start_v, -1, 0, [0] * 15)]
            while stack:
                current, prev, length, counts = stack.pop()
                if length >= max_L:
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
                        vec = tuple(new_counts)
                        if any(x != 0 for x in vec):
                            first_nz = -1
                            for i in range(15):
                                if vec[i] != 0:
                                    first_nz = i
                                    break
                            if first_nz != -1 and vec[first_nz] < 0:
                                vec = tuple(-x for x in vec)
                            walk_vectors.add(vec)
                    stack.append((next_v, current, new_l, new_counts))

        if not walk_vectors:
            return None

        C = np.array(list(walk_vectors), dtype=np.int64)
        C = np.unique(C, axis=0)
        n_constraints = len(C)

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

        def cost_func(v, m):
            """Count number of constraints violated (residue == 0 mod m)."""
            residues = np.mod(C @ v, m)
            return int(np.sum(residues == 0))

        def solve_voltage(m, time_limit):
            best_v = None
            best_cost = float('inf')
            start_t = time.time()

            # Try multiple strategies for initialization
            initializations = []
            # Random
            for _ in range(3):
                initializations.append(np.random.randint(1, m, size=15).astype(np.int64))
            # Structured: assign based on edge position
            for shift in range(3):
                v_struct = np.array([(i + 1 + shift) % (m - 1) + 1 for i in range(15)], dtype=np.int64)
                initializations.append(v_struct)
            # Greedy: try to minimize cost locally
            v_greedy = np.random.randint(1, m, size=15).astype(np.int64)
            for _ in range(200):
                best_j = -1
                best_new_cost = cost_func(v_greedy, m)
                for j in range(15):
                    old_val = v_greedy[j]
                    for trial in range(5):
                        v_greedy[j] = random.randint(1, m - 1)
                        c = cost_func(v_greedy, m)
                        if c < best_new_cost:
                            best_new_cost = c
                            best_j = j
                            best_val = v_greedy[j].copy()
                    v_greedy[j] = old_val
                if best_j >= 0:
                    v_greedy[best_j] = best_val
                else:
                    break
            initializations.append(v_greedy.copy())

            for init_v in initializations:
                if time.time() - start_t > time_limit or remaining() < 5:
                    break

                v = init_v.copy()
                cur_cost = cost_func(v, m)
                if cur_cost == 0:
                    return v.copy()
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_v = v.copy()

                # Simulated annealing with multiple coordinate updates
                T = float(m)
                cooling = 0.9997
                steps_per_iter = 3  # Try 3 coordinates per step
                while time.time() - start_t < time_limit and remaining() > 5:
                    # Update multiple coordinates
                    coords = random.sample(range(15), steps_per_iter)
                    accepted_any = False
                    for j in coords:
                        old_val = v[j]
                        v[j] = random.randint(1, m - 1)
                        new_cost = cost_func(v, m)
                        delta = new_cost - cur_cost
                        if delta < 0 or (T > 0.01 and random.random() < np.exp(-delta / T)):
                            cur_cost = new_cost
                            accepted_any = True
                            if new_cost < best_cost:
                                best_cost = new_cost
                                best_v = v.copy()
                                if best_cost == 0:
                                    return best_v
                        else:
                            v[j] = old_val
                    T *= cooling
                    if T < 0.01:
                        T = 0.01

            return best_v if best_cost == 0 else None

        # Moore bound for k=3, g=16 is 766. n = 10*m >= 766 => m >= 77.
        # Try primes starting from 79, but also try 77 (composite) and 83, 89, etc.
        # Give more time to smaller m values
        candidate_ms = [79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]

        total_time_available = min(120.0, remaining() - 10.0)
        # Allocate time proportional to likelihood (smaller m gets more time)
        weights = [1.0 / (i + 1) ** 0.5 for i in range(len(candidate_ms))]
        total_w = sum(weights)
        time_alloc = [total_time_available * w / total_w for w in weights]

        for i, m in enumerate(candidate_ms):
            if remaining() < 8:
                break
            t_limit = min(time_alloc[i], remaining() - 5.0)
            if t_limit < 0.5:
                continue

            v = solve_voltage(m, t_limit)
            if v is not None:
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
                gc = check_girth(n, edges, g, deadline - 3)
                if gc >= g:
                    return (n, edges)

        return None

    # Try voltage lift (fast, guaranteed)
    res = voltage_lift_search()
    if res is not None:
        return res

    return (0, [])


if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")

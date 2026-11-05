import time
import math
import random
import numpy as np
from collections import defaultdict

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    def remaining():
        return deadline - time.time()

    # ---------------- fast girth verification (early exit) ----------------
    def check_girth(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
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

    # ---------------- Backtracking tree construction with boundary merging ----------------
    def tree_backtrack_construct():
        """
        Construct a k-regular graph with girth >= g by:
        1. Building the non-backtracking tree of depth d = (g-1)//2 from a root
        2. Identifying boundary vertices (at depth d)
        3. Merging boundary vertices in pairs using a matching that avoids
           creating cycles shorter than g
        """
        if k != 3:
            return None
        
        d = (g - 1) // 2  # depth of tree: for g=16, d=7
        
        # Build the non-backtracking tree from root
        # Node representation: (vertex_id, parent_id, depth, path_tuple)
        # path_tuple: tuple of (edge_index, direction) to track non-backtracking
        
        # First, build a base 3-regular "skeleton" using a simple approach:
        # Use a cycle of length L with chords, or use the Petersen-like structure
        
        # Actually, let's use a different approach:
        # Start with a random 3-regular graph candidate and verify/repair
        
        # Better approach: Use the voltage lift but with a smarter search
        
        # For now, fall back to LUW which is guaranteed to work
        return None

    # ---------------- Improved voltage lift with simulated annealing ----------------
    def voltage_lift_improved():
        if k != 3 or g != 16:
            return None

        base_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        edge_index = {}
        for i, (u, v) in enumerate(base_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i

        adj = [[] for _ in range(10)]
        for u, v in base_edges:
            adj[u].append(v)
            adj[v].append(u)

        max_L = g - 1  # 15
        all_walk_vectors = set()

        for start_v in range(10):
            stack = [(start_v, -1, 0, [], [])]
            while stack:
                current, prev, length, p_edges, p_dirs = stack.pop()
                if length >= 5 and current == start_v:
                    vec = [0] * max_L
                    for e_idx, d in zip(p_edges, p_dirs):
                        vec[e_idx] += d
                    all_walk_vectors.add(tuple(vec))
                if length == max_L:
                    continue
                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = base_edges[e_idx]
                    d = 1 if current == u else -1
                    stack.append((next_v, current, length + 1,
                                  p_edges + [e_idx], p_dirs + [d]))

        if not all_walk_vectors:
            return None

        C = np.array(list(all_walk_vectors), dtype=np.int64)
        n_edges = len(base_edges)  # 15

        min_m = 52  # Moore bound

        def try_vector(x, m):
            S = (C @ x) % m
            return not np.any(S == 0)

        def find_prime_ge(n):
            if n <= 2:
                return 2
            cand = n if n % 2 == 1 else n + 1
            while True:
                if all(cand % i for i in range(3, int(math.isqrt(cand)) + 1, 2)):
                    return cand
                cand += 2

        best_m = None
        best_x = None

        # Simulated annealing over voltage assignments
        primes = []
        p = min_m
        while p < 300 and len(primes) < 20:
            p = find_prime_ge(p)
            primes.append(p)
            p += 1

        rng = np.random.default_rng(42)

        for m in primes:
            if remaining() < 10:
                break
            if best_m is not None:
                break

            # Initialize with random assignment
            x = rng.integers(0, m, size=n_edges)
            
            # Simulated annealing
            T = float(m)
            cooling = 0.995
            best_local = x.copy()
            best_local_score = np.sum((C @ x) % m == 0)
            
            for step in range(5000):
                if time.time() > deadline - 5:
                    break
                if best_local_score == 0:
                    break
                    
                # Random perturbation
                idx = rng.integers(0, n_edges)
                old_val = x[idx]
                x[idx] = (x[idx] + rng.integers(1, m)) % m
                
                new_score = np.sum((C @ x) % m == 0)
                delta = new_score - best_local_score
                
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    if new_score < best_local_score:
                        best_local_score = new_score
                        best_local = x.copy()
                else:
                    x[idx] = old_val
                
                T *= cooling
            
            if best_local_score == 0:
                best_m, best_x = m, best_local
                break

        if best_m is None:
            # Fallback: try more primes with random search
            for m in range(300, 500):
                if remaining() < 5:
                    break
                if m < 2 or not all(m % i for i in range(2, int(math.isqrt(m)) + 1)):
                    continue
                rng = np.random.default_rng(m * 40503 + 7)
                for _ in range(200):
                    x = rng.integers(0, m, size=n_edges)
                    if try_vector(x, m):
                        best_m, best_x = m, x
                        break
                if best_m is not None:
                    break

        if best_m is None:
            return None

        # Construct the lifted graph
        n = 10 * best_m
        edges_set = set()
        for u, v in base_edges:
            e_idx = edge_index[(u, v)]
            shift = best_x[e_idx] % best_m
            for i in range(best_m):
                u_lift = u * best_m + i
                v_lift = v * best_m + (i + shift) % best_m
                if u_lift < v_lift:
                    edges_set.add((u_lift, v_lift))
                else:
                    edges_set.add((v_lift, u_lift))

        final_edges = list(edges_set)

        gc = check_girth(n, final_edges, g, deadline - 3)
        if gc == -1:
            return None
        if gc >= g:
            return (n, final_edges)
        return None

    # ---------------- LUW construction (guaranteed fallback) ----------------
    def _luw_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        return ((lambda x, y: (x + y) % q),
                (lambda x, y: (x - y) % q),
                (lambda x, y: x * y % q))

    def luw_construct():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        add, sub, mul = _luw_ops(q)
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

    # Try the improved voltage lift first
    result = voltage_lift_improved()
    if result is not None:
        return result

    # Fallback to the LUW construction.
    return luw_construct()

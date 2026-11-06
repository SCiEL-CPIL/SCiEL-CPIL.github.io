import time
import sys
from random import Random
from collections import deque

def construct():
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]

    def fallback():
        out = []
        for spec in specs:
            edges = instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
        return out

    def compress(color):
        m = len(color)
        if m == 0:
            return []
        mapping = {}
        new = [0] * m
        nxt = 0
        for i, c in enumerate(color):
            if c not in mapping:
                mapping[c] = nxt
                nxt += 1
            new[i] = mapping[c]
        return new

    def solve_instance(n, d, seed, deadline):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Build conflict graph on edges
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)

        neigh = [[] for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                if a == c or a == dd or b == c or b == dd:
                    neigh[i].append(j)
                    neigh[j].append(i)
                else:
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        neigh[i].append(j)
                        neigh[j].append(i)

        deg = [len(neigh[i]) for i in range(m)]
        max_deg = max(deg) if deg else 0

        # Greedy initial coloring with K0 = max_deg + 1 (always valid for conflict graph)
        K0 = max_deg + 1
        order = sorted(range(m), key=lambda x: deg[x], reverse=True)
        color0 = [-1] * m
        for v in order:
            forbidden = set()
            for u in neigh[v]:
                if color0[u] != -1:
                    forbidden.add(color0[u])
            c = 0
            while c in forbidden:
                c += 1
            color0[v] = c

        # Local search: try to reduce colour count by merging
        # We'll maintain a colouring and attempt to remove one colour at a time
        best_color = color0[:]
        best_K = K0

        rng = Random(seed * 7919 + 42)

        def count_conflicts(color, K):
            """Count number of conflicting pairs in the colouring (should be 0 for valid)."""
            cnt = 0
            for i in range(m):
                for j in neigh[i]:
                    if j > i and color[i] == color[j]:
                        cnt += 1
            return cnt

        def local_search_reduce(color, K, time_limit):
            """
            Try to find a valid K-colouring starting from an initial assignment
            (which may have conflicts) using simulated annealing / min-conflicts.
            Returns (color, success).
            """
            if time.time() > time_limit:
                return color, False

            # Ensure colour values are in [0, K)
            for i in range(m):
                if color[i] >= K:
                    color[i] = rng.randrange(K)

            conflicts = count_conflicts(color, K)
            if conflicts == 0:
                return color, True

            best_color_ls = color[:]
            best_conflicts = conflicts

            # Simulated annealing parameters
            T = 1.0
            T_min = 0.01
            cooling = 0.995
            max_iters = 200000

            for it in range(max_iters):
                if time.time() > time_limit:
                    break
                if conflicts == 0:
                    return color, True

                # Pick a random vertex, reassign to a random colour in [0, K)
                v = rng.randrange(m)
                old_c = color[v]
                new_c = rng.randrange(K)
                if new_c == old_c:
                    continue

                # Compute delta conflicts
                # Remove old colour conflicts, add new colour conflicts
                delta = 0
                for u in neigh[v]:
                    if color[u] == old_c:
                        delta -= 1
                    if color[u] == new_c:
                        delta += 1

                color[v] = new_c
                conflicts += delta

                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_color_ls = color[:]
                elif conflicts < best_conflicts + 5 or T > T_min * 2:
                    # Accept with probability
                    import math
                    if delta <= 0 or rng.random() < math.exp(-delta / max(T, 0.001)):
                        pass
                    else:
                        # Reject
                        color[v] = old_c
                        conflicts -= delta
                else:
                    # Reject
                    color[v] = old_c
                    conflicts -= delta

                T *= cooling

            if best_conflicts == 0:
                return best_color_ls, True
            return color, False

        # Try to reduce K from K0 down to lower bound
        lower_bound = max_deg + 1  # Actually chromatic number >= max degree + 1 for conflict graph
        # But for our problem, the conflict graph may have lower chromatic number
        # Use a better lower bound: size of max clique in conflict graph is hard,
        # so use max degree + 1 as safe lower bound, but try to go lower

        current_color = color0[:]
        current_K = K0

        for target_K in range(current_K - 1, 2, -1):
            if time.time() > deadline - 5.0:
                break

            # Try multiple restarts of local search
            success = False
            for attempt in range(10):
                if time.time() > deadline - 5.0:
                    break
                # Start from current best colouring, randomise some colours
                trial = current_color[:]
                # Randomly perturb
                for i in range(m):
                    if rng.random() < 0.3:
                        trial[i] = rng.randrange(target_K)
                # Ensure colours are in range
                for i in range(m):
                    if trial[i] >= target_K:
                        trial[i] = rng.randrange(target_K)

                result, ok = local_search_reduce(trial, target_K, deadline - 5.0)
                if ok:
                    current_color = result
                    current_K = target_K
                    success = True
                    break

            if not success:
                # Try a few more with different initialisations
                for attempt in range(5):
                    if time.time() > deadline - 5.0:
                        break
                    trial = [rng.randrange(target_K) for _ in range(m)]
                    result, ok = local_search_reduce(trial, target_K, deadline - 5.0)
                    if ok:
                        current_color = result
                        current_K = target_K
                        success = True
                        break

            if not success:
                break  # Can't reduce further

        return compress(current_color)

    try:
        start = time.time()
        deadline = start + 130.0
        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = deadline - time.time()
            share = remaining / (len(specs) - idx)
            share = min(share, 40.0)
            share = max(share, 1.0)
            instance_deadline = start + (time.time() - start) + share
            out.append(solve_instance(n, d, seed, instance_deadline))
        return out
    except Exception:
        return fallback()

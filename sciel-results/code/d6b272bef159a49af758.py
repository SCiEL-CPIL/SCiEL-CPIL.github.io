import time
import sys
import math
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

        # DSATUR greedy coloring for better initial solution
        def dsatur_coloring():
            color = [-1] * m
            # Track saturation degree (number of different colors on neighbors)
            sat_degree = [0] * m
            # Track which colors are used on neighbors
            neighbor_colors = [set() for _ in range(m)]
            
            uncolored = list(range(m))
            
            for _ in range(m):
                # Find vertex with highest saturation degree (tie-break by degree)
                best_v = -1
                best_sat = -1
                best_deg = -1
                for v in uncolored:
                    sat = len(neighbor_colors[v])
                    if sat > best_sat or (sat == best_sat and deg[v] > best_deg):
                        best_v = v
                        best_sat = sat
                        best_deg = deg[v]
                        
                uncolored.remove(best_v)
                
                # Assign smallest available color
                forbidden = neighbor_colors[best_v]
                c = 0
                while c in forbidden:
                    c += 1
                color[best_v] = c
                
                # Update neighbors
                for u in neigh[best_v]:
                    if color[u] == -1:
                        if c not in neighbor_colors[u]:
                            neighbor_colors[u].add(c)
                        
            return color

        color0 = dsatur_coloring()
        K0 = max(color0) + 1

        rng = Random(seed * 7919 + 42)

        def count_conflicts(color):
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
            (which may have conflicts) using min-conflicts heuristic.
            Returns (color, success).
            """
            if time.time() > time_limit:
                return color, False

            # Ensure colour values are in [0, K)
            for i in range(m):
                if color[i] >= K:
                    color[i] = rng.randrange(K)

            conflicts = count_conflicts(color)
            if conflicts == 0:
                return color, True

            best_color_ls = color[:]
            best_conflicts = conflicts

            # Min-conflicts with simulated annealing
            T = 1.0
            T_min = 0.01
            cooling = 0.998
            max_iters = 500000
            stall_count = 0
            stall_limit = 5000

            for it in range(max_iters):
                if time.time() > time_limit:
                    break
                if conflicts == 0:
                    return color, True

                # Pick a random vertex that has a conflict (if possible)
                v = rng.randrange(m)
                old_c = color[v]
                
                # Count conflicts for this vertex with old color
                old_conflicts = 0
                for u in neigh[v]:
                    if color[u] == old_c:
                        old_conflicts += 1
                
                if old_conflicts == 0:
                    # This vertex has no conflicts, pick another
                    continue

                # Try all colors and find the one with minimum conflicts
                best_new_c = old_c
                best_new_conflicts = old_conflicts
                for new_c in range(K):
                    if new_c == old_c:
                        continue
                    new_conflicts = 0
                    for u in neigh[v]:
                        if color[u] == new_c:
                            new_conflicts += 1
                    if new_conflicts < best_new_conflicts:
                        best_new_conflicts = new_conflicts
                        best_new_c = new_c

                new_c = best_new_c
                delta = best_new_conflicts - old_conflicts

                color[v] = new_c
                conflicts += delta

                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_color_ls = color[:]
                    stall_count = 0
                else:
                    stall_count += 1

                # Accept with probability based on simulated annealing
                if delta > 0:
                    prob = math.exp(-delta / max(T, 0.001))
                    if rng.random() > prob:
                        # Reject
                        color[v] = old_c
                        conflicts -= delta

                T *= cooling

                # Restart if stalled
                if stall_count > stall_limit:
                    # Randomly reset some vertices
                    for i in range(m):
                        if rng.random() < 0.5:
                            color[i] = rng.randrange(K)
                    conflicts = count_conflicts(color)
                    stall_count = 0
                    T = 1.0

            if best_conflicts == 0:
                return best_color_ls, True
            return color, False

        current_color = color0[:]
        current_K = K0

        # Try to reduce K from K0 down to lower bound
        # Use a better lower bound based on maximum degree in conflict graph
        # The chromatic number is at least max_degree + 1 for the conflict graph
        # But we can often do better
        for target_K in range(current_K - 1, 2, -1):
            if time.time() > deadline - 5.0:
                break

            # Try multiple restarts of local search
            success = False
            num_restarts = 15
            for attempt in range(num_restarts):
                if time.time() > deadline - 5.0:
                    break
                
                # Start from current best colouring, randomise some colours
                trial = current_color[:]
                
                # Randomly perturb
                perturbation_rate = 0.3 + 0.3 * (attempt / num_restarts)
                for i in range(m):
                    if rng.random() < perturbation_rate:
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

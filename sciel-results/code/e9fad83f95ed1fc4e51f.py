import time
import math
import numpy as np
from random import Random

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

        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)

        # Build conflict graph: two edges conflict if they share a vertex or are joined by an edge
        neigh = [set() for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                if a == c or a == dd or b == c or b == dd:
                    neigh[i].add(j)
                    neigh[j].add(i)
                else:
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        neigh[i].add(j)
                        neigh[j].add(i)

        # Convert to sorted lists for speed
        neigh_list = [sorted(s) for s in neigh]
        deg = np.array([len(x) for x in neigh_list], dtype=np.int32)

        # Maximum Independent Set via branch-and-bound
        # We want to find the largest subset of edges that form an induced matching
        def find_mis(available, best_size, deadline_check):
            """Branch and bound to find maximum independent set in available set."""
            if len(available) == 0:
                return 0, []
            if len(available) <= best_size:
                return 0, []
            if deadline_check():
                return 0, []

            # Pick vertex with minimum degree in the induced subgraph
            # Use a heuristic: pick vertex with fewest neighbors in available
            avail_set = set(available)
            min_deg = len(available)
            pivot = available[0]
            for v in available:
                cnt = sum(1 for u in available if u != v and u in neigh_set[v])
                if cnt < min_deg:
                    min_deg = cnt
                    pivot = v
                if min_deg == 0:
                    break

            # Branch 1: include pivot
            # Neighbors of pivot in available cannot be included
            piv_neigh = neigh_set[pivot] & avail_set
            avail_without_piv_neigh = [v for v in available if v != pivot and v not in piv_neigh]
            # The set {pivot} ∪ MIS(avail_without_piv_neigh) is an independent set
            # But we need to be careful: removing pivot's neighbors is correct
            size1, set1 = find_mis(avail_without_piv_neigh, best_size, deadline_check)
            size1 += 1
            set1.append(pivot)

            # Branch 2: exclude pivot
            avail_without_pivot = [v for v in available if v != pivot]
            size2, set2 = find_mis(avail_without_pivot, best_size, deadline_check)

            if size1 >= size2:
                return size1, set1
            else:
                return size2, set2

        # Precompute neighbor sets for fast lookup
        neigh_set = [set(x) for x in neigh_list]

        # Greedy coloring to get initial upper bound
        def greedy_color():
            color = [-1] * m
            uncolored = list(range(m))
            while uncolored:
                c = 0
                placed = []
                while uncolored:
                    remaining = []
                    for v in uncolored:
                        # Check if v can have color c
                        can = True
                        for u in neigh_list[v]:
                            if color[u] == c:
                                can = False
                                break
                        if can:
                            color[v] = c
                            placed.append(v)
                        else:
                            remaining.append(v)
                    uncolored = remaining
                    if not placed:
                        c += 1
                        placed = []
                # Reset: this simple greedy doesn't work well, use proper one
            return color

        def greedy_coloring():
            """Standard greedy coloring with DSATUR-like ordering."""
            color = [-1] * m
            uncolored = set(range(m))
            color_count = 0
            while uncolored:
                # Pick vertex with highest saturation (most different colored neighbors)
                best_v = None
                best_sat = -1
                best_deg = -1
                for v in uncolored:
                    sat = len(set(color[u] for u in neigh_list[v] if color[u] != -1))
                    if sat > best_sat or (sat == best_sat and len(neigh_list[v]) > best_deg):
                        best_sat = sat
                        best_deg = len(neigh_list[v])
                        best_v = v
                # Assign smallest available color
                forbidden = set(color[u] for u in neigh_list[best_v] if color[u] != -1)
                c = 0
                while c in forbidden:
                    c += 1
                color[best_v] = c
                uncolored.discard(best_v)
            return color

        color0 = greedy_coloring()
        K0 = max(color0) + 1 if m > 0 else 0

        # Now use MIS extraction to build coloring
        # Strategy: repeatedly extract a large independent set, assign it a new color
        def build_coloring_mis(deadline):
            color = [-1] * m
            uncolored = list(range(m))
            color_idx = 0
            while uncolored:
                if time.time() > deadline:
                    break
                # Find a large independent set among uncolored
                # Use greedy + local improvement for speed
                # Greedy: repeatedly pick vertex with min degree in remaining
                available = list(uncolored)
                available_set = set(available)
                chosen = []
                while available:
                    # Pick vertex with fewest neighbors in available
                    min_deg = len(available)
                    pivot = available[0]
                    for v in available:
                        cnt = 0
                        for u in neigh_list[v]:
                            if u in available_set and u != v:
                                cnt += 1
                        if cnt < min_deg:
                            min_deg = cnt
                            pivot = v
                        if min_deg == 0:
                            break
                    chosen.append(pivot)
                    # Remove pivot and its neighbors from available
                    piv_neigh = neigh_set[pivot]
                    available = [v for v in available if v != pivot and v not in piv_neigh]
                    available_set = set(available)

                # Try to improve: add back some vertices that were excluded
                # For each excluded vertex, check if it can be added
                excluded = [v for v in uncolored if v not in set(chosen)]
                for v in excluded:
                    if time.time() > deadline:
                        break
                    can_add = True
                    for u in chosen:
                        if u in neigh_set[v]:
                            can_add = False
                            break
                    if can_add:
                        chosen.append(v)

                # Assign color
                for v in chosen:
                    color[v] = color_idx
                uncolored = [v for v in uncolored if v not in set(chosen)]
                color_idx += 1

            # Fill in any remaining uncolored with greedy
            if uncolored:
                for v in uncolored:
                    if color[v] == -1:
                        forbidden = set(color[u] for u in neigh_list[v] if color[u] != -1)
                        c = 0
                        while c in forbidden:
                            c += 1
                        color[v] = c
            return color

        # Build initial coloring via MIS extraction
        color_mis = build_coloring_mis(deadline - 2.0)
        K_mis = max(color_mis) + 1 if m > 0 else 0

        # Use the better of greedy and MIS
        if K_mis < K0:
            best_color = color_mis
            best_K = K_mis
        else:
            best_color = color0
            best_K = K0

        # Now try to reduce K further via local search / simulated annealing
        # Build conflict count vector
        neigh_arr = [np.array(x, dtype=np.int32) for x in neigh_list]

        def count_conflicts(color_arr):
            total = 0
            for i in range(m):
                if neigh_arr[i].size > 0:
                    total += int((color_arr[neigh_arr[i]] == color_arr[i]).sum())
            return total // 2

        def anneal_reduce(color_init, K, time_limit, rng):
            """Try to find valid K-coloring via simulated annealing."""
            if K <= 0:
                return None
            color_arr = np.array(color_init, dtype=np.int32)
            color_arr[color_arr >= K] = np.random.randint(0, K, size=int((color_arr >= K).sum()))

            conflicts = count_conflicts(color_arr)
            if conflicts == 0:
                return color_arr.tolist()

            T = float(K) * 0.5
            T_min = 0.005
            cooling = 0.999
            best_color = color_arr.copy()
            best_conflicts = conflicts
            no_improve = 0
            max_no_improve = 5000

            while time.time() < time_limit:
                if conflicts == 0:
                    return color_arr.tolist()
                if no_improve > max_no_improve:
                    break

                v = rng.randrange(m)
                new_c = rng.randrange(K)
                old_c = int(color_arr[v])
                if new_c == old_c:
                    continue

                neighbors = neigh_arr[v]
                if neighbors.size == 0:
                    delta = 0
                else:
                    delta = int((color_arr[neighbors] == new_c).sum()) - int((color_arr[neighbors] == old_c).sum())

                color_arr[v] = new_c
                conflicts += delta

                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_color = color_arr.copy()
                    no_improve = 0
                    if best_conflicts == 0:
                        return best_color.tolist()
                elif delta > 0:
                    if T > T_min:
                        if rng.random() < math.exp(-delta / max(T, 1e-9)):
                            pass
                        else:
                            color_arr[v] = old_c
                            conflicts -= delta
                    else:
                        color_arr[v] = old_c
                        conflicts -= delta
                    no_improve += 1
                else:
                    no_improve += 1

                T = max(T * cooling, T_min)

            if best_conflicts == 0:
                return best_color.tolist()
            return None

        # Try to reduce K step by step
        rng = Random(seed * 7919 + 42)
        lower_K = max(3, int(deg.max())) if m > 0 else 0

        for target_K in range(best_K, max(lower_K - 1, 2), -1):
            if time.time() > deadline - 3.0:
                break
            success = False
            max_restarts = 10
            for attempt in range(max_restarts):
                if time.time() > deadline - 2.0:
                    break
                if attempt == 0:
                    trial = best_color[:]
                elif attempt < 4:
                    trial = best_color[:]
                    for i in range(m):
                        if trial[i] >= target_K:
                            trial[i] = rng.randrange(target_K)
                        elif rng.random() < 0.3:
                            trial[i] = rng.randrange(target_K)
                else:
                    trial = [rng.randrange(target_K) for _ in range(m)]

                remaining = deadline - 2.0 - time.time()
                anneal_time = time.time() + max(0.5, remaining / (max_restarts - attempt) * 0.7)
                result = anneal_reduce(trial, target_K, anneal_time, rng)
                if result is not None:
                    best_color = result
                    best_K = target_K
                    success = True
                    break

            if not success:
                break

        return compress(best_color)

    try:
        start = time.time()
        total_budget = 140.0
        per_instance = total_budget / len(specs)

        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = start + total_budget - time.time()
            share = min(per_instance, remaining / (len(specs) - idx))
            share = max(share, 5.0)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))

        return out
    except Exception:
        out = []
        for spec in specs:
            n, d, seed = spec
            edges = instance(n, d, seed)
            if not edges:
                out.append([])
                continue
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj.get(x, set())}
                taken = set()
                for f, c in colour.items():
                    if set(f) & near:
                        taken.add(c)
                c = 0
                while c in taken:
                    c += 1
                colour[e] = c
            out.append([colour[e] for e in edges])
        return out

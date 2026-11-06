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
        neigh_set = [set(x) for x in neigh_list]
        
        # Bitset representation for fast operations
        # Using Python integers as bitsets
        bit_neigh = [0] * m
        for i in range(m):
            mask = 0
            for j in neigh_list[i]:
                mask |= (1 << j)
            bit_neigh[i] = mask

        # Greedy coloring to get initial upper bound
        def greedy_coloring():
            """Standard greedy coloring with DSATUR-like ordering."""
            color = [-1] * m
            uncolored = set(range(m))
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

        # Function to find a large independent set (induced matching) in a subset of vertices
        # Uses a greedy heuristic with bitsets for speed
        def find_large_is(subset_bits, time_limit):
            """
            Find a large independent set from the vertices indicated by subset_bits.
            Returns a list of vertex indices.
            """
            current_bits = subset_bits
            chosen = []
            
            while current_bits:
                if time.time() > time_limit:
                    break
                
                # Find vertex with minimum degree in the current subgraph
                best_v = -1
                best_deg = m + 1
                
                # Iterate through set bits
                temp = current_bits
                while temp:
                    # Get lowest set bit
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    # Count neighbors in current subgraph
                    deg = (bit_neigh[v] & current_bits).bit_count()
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                    if best_deg == 0:
                        break
                    temp ^= lsb
                
                if best_v == -1:
                    break
                    
                chosen.append(best_v)
                # Remove v and its neighbors
                current_bits &= ~(bit_neigh[best_v] | (1 << best_v))
                
            return chosen

        # Build coloring by repeatedly extracting large independent sets
        def build_coloring_mis(time_limit):
            color = [-1] * m
            uncolored_bits = (1 << m) - 1
            color_idx = 0
            
            while uncolored_bits:
                if time.time() > time_limit:
                    break
                
                # Find large IS
                is_set = find_large_is(uncolored_bits, time_limit)
                
                # Assign color
                mask = 0
                for v in is_set:
                    color[v] = color_idx
                    mask |= (1 << v)
                
                uncolored_bits &= ~mask
                color_idx += 1
            
            # Fill remaining
            if uncolored_bits:
                # Simple greedy for leftovers
                temp = uncolored_bits
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    forbidden = set(color[u] for u in neigh_list[v] if color[u] != -1)
                    c = 0
                    while c in forbidden:
                        c += 1
                    color[v] = c
                    temp ^= lsb
            
            return color

        # Try MIS extraction
        color_mis = build_coloring_mis(deadline - 2.0)
        K_mis = max(color_mis) + 1 if m > 0 else 0

        if K_mis < K0:
            best_color = color_mis
            best_K = K_mis
        else:
            best_color = color0
            best_K = K0

        # Local Search: Simulated Annealing to reduce K
        def count_conflicts(color_arr):
            total = 0
            for i in range(m):
                if neigh_list[i]:
                    total += sum(1 for u in neigh_list[i] if color_arr[u] == color_arr[i])
            return total // 2

        def anneal_reduce(color_init, K, time_limit, rng):
            if K <= 0:
                return None
            color_arr = list(color_init)
            # Clamp colors
            for i in range(m):
                if color_arr[i] >= K:
                    color_arr[i] = rng.randrange(K)

            conflicts = count_conflicts(color_arr)
            if conflicts == 0:
                return color_arr

            T = float(K) * 0.5
            T_min = 0.005
            cooling = 0.999
            best_color = color_arr[:]
            best_conflicts = conflicts
            no_improve = 0
            max_no_improve = 5000

            while time.time() < time_limit:
                if conflicts == 0:
                    return color_arr
                if no_improve > max_no_improve:
                    break

                v = rng.randrange(m)
                new_c = rng.randrange(K)
                old_c = color_arr[v]
                if new_c == old_c:
                    continue

                # Calculate delta
                delta = 0
                for u in neigh_list[v]:
                    if color_arr[u] == new_c:
                        delta += 1
                    if color_arr[u] == old_c:
                        delta -= 1

                color_arr[v] = new_c
                conflicts += delta

                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_color = color_arr[:]
                    no_improve = 0
                    if best_conflicts == 0:
                        return best_color
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
                return best_color
            return None

        # Try to reduce K
        rng = Random(seed * 7919 + 42)
        lower_K = max(3, max(len(x) for x in neigh_list) + 1) if m > 0 else 0

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

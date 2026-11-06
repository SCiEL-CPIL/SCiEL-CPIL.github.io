import time
import sys
from random import Random
import math

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

        neigh_list = [list(s) for s in neigh]
        deg = [len(neigh_list[i]) for i in range(m)]

        # DSATUR coloring for initial solution
        def dsatur_coloring():
            color = [-1] * m
            color_sets = [set() for _ in range(m)]
            uncolored = list(range(m))
            
            while uncolored:
                best_v = -1
                best_sat = -1
                best_deg = -1
                for v in uncolored:
                    sat = len(color_sets[v])
                    dv = len(neigh_list[v])
                    if sat > best_sat or (sat == best_sat and dv > best_deg):
                        best_sat = sat
                        best_deg = dv
                        best_v = v
                
                c = 0
                while c in color_sets[best_v]:
                    c += 1
                color[best_v] = c
                
                for u in neigh_list[best_v]:
                    if color[u] == -1:
                        color_sets[u].add(c)
                
                uncolored.remove(best_v)
            
            return color

        color0 = dsatur_coloring()
        K0 = max(color0) + 1

        rng = Random(seed * 7919 + 42)

        def count_conflicts(color):
            cnt = 0
            for i in range(m):
                for j in neigh_list[i]:
                    if j > i and color[i] == color[j]:
                        cnt += 1
            return cnt

        def local_search_reduce(color, K, time_limit):
            if time.time() > time_limit:
                return color, False

            for i in range(m):
                if color[i] >= K:
                    color[i] = rng.randrange(K)

            conflicts = count_conflicts(color)
            if conflicts == 0:
                return color, True

            best_color_ls = color[:]
            best_conflicts = conflicts

            T = 1.0
            cooling = 0.9995
            max_iters = 500000

            for it in range(max_iters):
                if time.time() > time_limit:
                    break
                if conflicts == 0:
                    return color, True

                v = rng.randrange(m)
                old_c = color[v]
                new_c = rng.randrange(K)
                if new_c == old_c:
                    continue

                delta = 0
                for u in neigh_list[v]:
                    if color[u] == old_c:
                        delta -= 1
                    if color[u] == new_c:
                        delta += 1

                color[v] = new_c
                conflicts += delta

                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_color_ls = color[:]
                elif delta <= 0 or rng.random() < math.exp(-delta / max(T, 0.001)):
                    pass
                else:
                    color[v] = old_c
                    conflicts -= delta

                T *= cooling

            if best_conflicts == 0:
                return best_color_ls, True
            return color, False

        def merge_color_pairs(color, K, time_limit):
            """Try to merge pairs of colors by finding if their union can be recolored into one color."""
            if time.time() > time_limit:
                return None, False
            
            # Group vertices by color
            color_groups = {}
            for i in range(m):
                c = color[i]
                if c not in color_groups:
                    color_groups[c] = []
                color_groups[c].append(i)
            
            colors_used = sorted(color_groups.keys())
            
            # Try merging pairs of colors
            for idx_i in range(len(colors_used)):
                if time.time() > time_limit - 1:
                    break
                for idx_j in range(idx_i + 1, len(colors_used)):
                    if time.time() > time_limit - 1:
                        break
                    
                    ci = colors_used[idx_i]
                    cj = colors_used[idx_j]
                    
                    # Get vertices in these two color groups
                    group_i = set(color_groups[ci])
                    group_j = set(color_groups[cj])
                    all_in_groups = group_i | group_j
                    
                    # Check if we can recolor all vertices in these groups with just color ci
                    # This is equivalent to checking if the induced subgraph on these vertices
                    # has a proper 1-coloring (i.e., is an independent set)
                    
                    # Actually, we need to check if we can recolor them using only color ci
                    # without creating conflicts with other vertices of color ci outside these groups
                    
                    # For each vertex in group_j, check if it can be moved to color ci
                    # without conflicting with existing color ci vertices
                    movable = True
                    for v in group_j:
                        # Check if v conflicts with any vertex in group_i
                        for u in group_i:
                            if u in neigh_list[v]:
                                movable = False
                                break
                        if not movable:
                            break
                    
                    if not movable:
                        continue
                    
                    # Try to move all of group_j to color ci
                    new_color = color[:]
                    for v in group_j:
                        new_color[v] = ci
                    
                    # Check if this creates any new conflicts
                    if count_conflicts(new_color) == 0:
                        # Compress
                        compressed = compress(new_color)
                        return compressed, True
            
            return None, False

        current_color = color0[:]
        current_K = K0

        for target_K in range(current_K - 1, 2, -1):
            if time.time() > deadline - 3.0:
                break

            success = False
            
            # First try pair merging
            result, ok = merge_color_pairs(current_color, current_K, deadline - 3.0)
            if ok and max(result) + 1 <= target_K:
                current_color = result
                current_K = max(result) + 1
                success = True
            
            if not success:
                for attempt in range(12):
                    if time.time() > deadline - 3.0:
                        break
                    trial = current_color[:]
                    for i in range(m):
                        if trial[i] >= target_K:
                            trial[i] = rng.randrange(target_K)
                        elif rng.random() < 0.2:
                            trial[i] = rng.randrange(target_K)

                    result, ok = local_search_reduce(trial, target_K, deadline - 3.0)
                    if ok:
                        current_color = result
                        current_K = target_K
                        success = True
                        break

            if not success:
                for attempt in range(6):
                    if time.time() > deadline - 3.0:
                        break
                    trial = [rng.randrange(target_K) for _ in range(m)]
                    result, ok = local_search_reduce(trial, target_K, deadline - 3.0)
                    if ok:
                        current_color = result
                        current_K = target_K
                        success = True
                        break

            if not success:
                break

        return compress(current_color)

    try:
        start = time.time()
        deadline = start + 145.0
        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = deadline - time.time()
            share = remaining / (len(specs) - idx)
            share = min(share, 50.0)
            share = max(share, 1.0)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))
        return out
    except Exception:
        return fallback()

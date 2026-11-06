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

        # Build adjacency of original graph
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)

        # Build conflict graph: two edges conflict if they share a vertex OR are joined by an edge
        # Edge i = (a, b), edge j = (c, d)
        # Conflict if: a==c, a==d, b==c, b==d (share vertex)
        # OR: c in adj_orig[a], d in adj_orig[a], c in adj_orig[b], d in adj_orig[b] (joined by edge)
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

        # DSATUR greedy coloring for initial solution
        color = [-1] * m
        sat = [0] * m  # saturation degree (number of different colors in neighbors)
        # Track which colors are used in neighbors
        neighbor_colors = [set() for _ in range(m)]
        
        # Order vertices by degree descending, then by some heuristic
        order = sorted(range(m), key=lambda i: -deg[i])
        
        for v in order:
            # Find available colors
            used = set()
            for u in neigh[v]:
                if color[u] != -1:
                    used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
        
        current_K = max(color) + 1 if color else 0
        best_color = color[:]
        
        # Deterministic local search to reduce conflicts and then reduce K
        def get_conflict_count(col, K):
            """Count total conflicts (pairs of adjacent vertices with same color)"""
            cnt = 0
            for i in range(m):
                ci = col[i]
                for j in neigh[i]:
                    if j > i and col[j] == ci:
                        cnt += 1
            return cnt

        def local_search_deterministic(col, K, time_limit, max_rounds=50):
            """Deterministic local search: repeatedly reassign the worst vertex"""
            if time.time() > time_limit:
                return col, False
            
            # Compute initial conflict count
            conflict_count = [0] * m
            for i in range(m):
                ci = col[i]
                for j in neigh[i]:
                    if col[j] == ci:
                        conflict_count[i] += 1
            total_conflicts = sum(conflict_count) // 2
            
            if total_conflicts == 0:
                return col, True
            
            best_col = col[:]
            best_total = total_conflicts
            
            for rnd in range(max_rounds):
                if time.time() > time_limit:
                    break
                if total_conflicts == 0:
                    return col, True
                
                # Find vertex with maximum conflict
                max_conf = -1
                worst_v = -1
                for i in range(m):
                    if conflict_count[i] > max_conf:
                        max_conf = conflict_count[i]
                        worst_v = i
                
                if max_conf == 0:
                    break
                
                # Try all possible colors for worst_v
                best_new_c = col[worst_v]
                best_delta = 0
                
                for new_c in range(K):
                    if new_c == col[worst_v]:
                        continue
                    # Compute delta
                    delta = 0
                    for u in neigh[worst_v]:
                        if col[u] == col[worst_v]:
                            delta -= 1
                        if col[u] == new_c:
                            delta += 1
                    if delta < best_delta:
                        best_delta = delta
                        best_new_c = new_c
                
                if best_delta < 0:
                    # Apply the best move
                    old_c = col[worst_v]
                    col[worst_v] = best_new_c
                    total_conflicts += best_delta
                    conflict_count[worst_v] += best_delta
                    for u in neigh[worst_v]:
                        conflict_count[u] -= 1  # was in conflict with old color
                        if col[u] == best_new_c:
                            conflict_count[u] += 1  # now in conflict with new color
                    
                    if total_conflicts < best_total:
                        best_total = total_conflicts
                        best_col = col[:]
                else:
                    # No improvement, try a random restart or break
                    break
            
            if best_total == 0:
                return best_col, True
            return best_col, False

        # First, ensure we have a valid coloring with current_K
        # The DSATUR coloring should already be valid, but let's verify
        # and fix if needed
        result, ok = local_search_deterministic(best_color, current_K, deadline - 5.0, max_rounds=100)
        best_color = result
        
        # Now try to reduce K
        for target_K in range(current_K - 1, 2, -1):
            if time.time() > deadline - 5.0:
                break
            
            success = False
            # Try multiple random initializations
            for attempt in range(30):
                if time.time() > deadline - 5.0:
                    break
                
                # Create a random coloring with target_K colors
                rng = Random(seed * 1000 + attempt)
                trial = [rng.randrange(target_K) for _ in range(m)]
                
                result, ok = local_search_deterministic(trial, target_K, deadline - 5.0, max_rounds=200)
                if ok:
                    best_color = result
                    current_K = target_K
                    success = True
                    break
            
            if not success:
                # Try one more approach: start from best_color and randomly reassign some vertices
                rng = Random(seed * 2000)
                trial = best_color[:]
                for i in range(m):
                    if trial[i] >= target_K:
                        trial[i] = rng.randrange(target_K)
                    elif rng.random() < 0.3:
                        trial[i] = rng.randrange(target_K)
                
                result, ok = local_search_deterministic(trial, target_K, deadline - 5.0, max_rounds=200)
                if ok:
                    best_color = result
                    current_K = target_K
                    success = True
                else:
                    break
        
        return compress(best_color)

    try:
        start = time.time()
        deadline = start + 140.0
        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = deadline - time.time()
            share = remaining / (len(specs) - idx)
            share = min(share, 45.0)
            share = max(share, 1.0)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))
        return out
    except Exception:
        return fallback()

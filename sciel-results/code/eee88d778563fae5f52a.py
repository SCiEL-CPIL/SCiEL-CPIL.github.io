import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(n, d, seed, time_limit):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Precompute adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Precompute conflict sets: two edges conflict if they share a vertex
        # or are joined by an edge (induced matching condition)
        conflict_sets = [set() for _ in range(m)]
        for i in range(m):
            ui, vi = edges[i]
            for j in range(i + 1, m):
                uj, vj = edges[j]
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict_sets[i].add(j)
                    conflict_sets[j].add(i)
                else:
                    if ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                        conflict_sets[i].add(j)
                        conflict_sets[j].add(i)

        static_deg = [len(conflict_sets[i]) for i in range(m)]

        # DSATUR greedy initialization
        def greedy_dsatur():
            colors = [-1] * m
            uncolored = set(range(m))
            # Precompute neighbor colors incrementally
            while uncolored:
                best_i, best_score = None, (-1, -1, -1)
                for i in uncolored:
                    neighbor_colors = set()
                    for j in conflict_sets[i]:
                        if colors[j] != -1:
                            neighbor_colors.add(colors[j])
                    sat = len(neighbor_colors)
                    deg = sum(1 for j in conflict_sets[i] if j in uncolored)
                    score = (sat, deg, static_deg[i])
                    if score > best_score:
                        best_score = score
                        best_i = i
                i = best_i
                taken = set()
                for j in conflict_sets[i]:
                    if colors[j] != -1:
                        taken.add(colors[j])
                c = 0
                while c in taken:
                    c += 1
                colors[i] = c
                uncolored.remove(i)
            return colors

        best_colors = greedy_dsatur()
        best_num = max(best_colors) + 1 if best_colors else 0

        # Local search: try to reduce colors by recoloring highest color class
        def try_reduce_colors(colors, target_K, time_budget):
            """Try to reduce coloring to use at most target_K colors."""
            current = colors[:]
            cur_max = max(current) if current else -1

            if cur_max < target_K:
                return current

            # Try to eliminate colors from cur_max down to target_K
            for target_max in range(cur_max, target_K - 1, -1):
                if time.time() > time_budget:
                    break

                to_recolor = [i for i in range(m) if current[i] == target_max]
                if not to_recolor:
                    continue

                success = False
                for attempt in range(100):
                    if time.time() > time_budget:
                        break
                    order = to_recolor[:]
                    random.shuffle(order)

                    temp = current[:]
                    valid = True
                    for i in order:
                        taken = set()
                        for j in conflict_sets[i]:
                            if temp[j] != -1 and temp[j] < target_max:
                                taken.add(temp[j])
                        avail = [c for c in range(target_max) if c not in taken]
                        if avail:
                            temp[i] = random.choice(avail)
                        else:
                            valid = False
                            break
                    if valid:
                        # Verify full validity
                        ok = True
                        for i in range(m):
                            ci = temp[i]
                            for j in conflict_sets[i]:
                                if temp[j] == ci:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            current = temp[:]
                            success = True
                            break
                if not success:
                    break

            return current

        # Apply local search to reduce colors
        if best_num > 1:
            for target in range(best_num - 1, 0, -1):
                if time.time() > deadline - 5.0:
                    break
                reduced = try_reduce_colors(best_colors, target, deadline - 5.0)
                if reduced is not None:
                    new_max = max(reduced) + 1 if reduced else 0
                    if new_max < best_num:
                        best_colors = reduced
                        best_num = new_max
                    else:
                        break

        # Additional local search: random restarts with perturbation
        best_num_final = best_num
        if time.time() < deadline - 10.0:
            for restart in range(20):
                if time.time() > deadline - 5.0:
                    break
                # Perturb: randomly recolor some edges
                perturbed = best_colors[:]
                num_perturb = max(1, m // 10)
                for _ in range(num_perturb):
                    idx = random.randint(0, m - 1)
                    # Try to recolor idx to a random valid color
                    taken = set()
                    for j in conflict_sets[idx]:
                        taken.add(perturbed[j])
                    avail = [c for c in range(best_num_final) if c not in taken]
                    if avail:
                        perturbed[idx] = random.choice(avail)

                # Try to reduce from perturbed
                for target in range(best_num_final - 1, 0, -1):
                    if time.time() > deadline - 5.0:
                        break
                    reduced = try_reduce_colors(perturbed, target, deadline - 5.0)
                    if reduced is not None:
                        new_max = max(reduced) + 1 if reduced else 0
                        if new_max < best_num_final:
                            best_colors = reduced
                            best_num_final = new_max
                        break

        return best_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        remaining = deadline - time.time()
        if remaining <= 0:
            # Fallback: simple greedy
            edges = instance(n, d, seed)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colors = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colors.items() if set(f) & near}
                c = 0
                while c in taken:
                    c += 1
                colors[e] = c
            results.append([colors[e] for e in edges])
            continue

        if n == 40:
            time_limit = min(30.0, remaining - 2.0)
        elif n == 60:
            time_limit = min(35.0, remaining - 2.0)
        else:
            time_limit = min(40.0, remaining - 2.0)

        result = solve_instance(n, d, seed, time_limit)
        results.append(result)

    return results

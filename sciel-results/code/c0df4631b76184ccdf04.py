import time
import sys
import math
from random import Random


def construct():
    def bitset_greedy_color(m, neigh, deg, rng=None, rand_order=False):
        """Faster greedy coloring using bitsets for neighbor tracking."""
        color = [-1] * m
        # For each color, track which vertices have it
        color_members = {}  # color -> set of vertices
        remaining = list(range(m))
        # Precompute neighbor bitsets
        neigh_bits = [0] * m
        for i in range(m):
            bits = 0
            for j in neigh[i]:
                bits |= (1 << j)
            neigh_bits[i] = bits

        for _ in range(m):
            if not remaining:
                break
            best = -1
            best_key = None
            for v in remaining:
                if rand_order and rng is not None:
                    # Count saturated colors
                    sat_count = 0
                    for c in color_members:
                        if any((1 << u) & neigh_bits[v] for u in color_members[c]):
                            sat_count += 1
                    key = (sat_count, deg[v], rng.random())
                else:
                    sat_count = 0
                    for c in color_members:
                        if any((1 << u) & neigh_bits[v] for u in color_members[c]):
                            sat_count += 1
                    key = (sat_count, deg[v])
                if best_key is None or key > best_key:
                    best_key = key
                    best = v
            # Find smallest available color
            c = 0
            while True:
                if c not in color_members:
                    break
                # Check if any member of color c is a neighbor
                conflict = False
                for u in color_members[c]:
                    if (1 << u) & neigh_bits[best]:
                        conflict = True
                        break
                if not conflict:
                    break
                c += 1
            color[best] = c
            if c not in color_members:
                color_members[c] = set()
            color_members[c].add(best)
            remaining.remove(best)
        return color, (max(color) + 1 if m else 0)
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

    def dsatur_color(m, neigh, deg, rng=None, rand_order=False):
        color = [-1] * m
        sat = [set() for _ in range(m)]
        remaining = list(range(m))
        for _ in range(m):
            best = -1
            best_key = None
            for v in remaining:
                if rand_order and rng is not None:
                    key = (len(sat[v]), deg[v], rng.random())
                else:
                    key = (len(sat[v]), deg[v])
                if best_key is None or key > best_key:
                    best_key = key
                    best = v
            c = 0
            while c in sat[best]:
                c += 1
            color[best] = c
            remaining.remove(best)
            for u in neigh[best]:
                if color[u] == -1:
                    sat[u].add(c)
        return color, (max(color) + 1 if m else 0)

    def greedy_clique(m, neigh):
        neighset = [set(s) for s in neigh]
        cand = list(range(m))
        clique = 0
        while cand:
            # pick vertex with max degree inside cand
            best = cand[0]
            best_d = -1
            bs = neighset[best]
            for x in cand:
                s = neighset[x]
                dd = sum(1 for y in cand if y in s)
                if dd > best_d:
                    best_d = dd
                    best = x
                    bs = s
            clique += 1
            keep = bs
            cand = [x for x in cand if x != best and x in keep]
        return clique

    def recompute_conflict(color, neigh, m):
        conflict = [0] * m
        for v in range(m):
            cv = color[v]
            s = 0
            for u in neigh[v]:
                if color[u] == cv:
                    s += 1
            conflict[v] = s
        return conflict, sum(conflict) // 2

    def local_search(m, neigh, color_init, K, rng, time_limit):
        if K <= 0:
            return color_init, False
        color = [c if c < K else rng.randrange(K) for c in color_init]
        conflict, total = recompute_conflict(color, neigh, m)
        if total == 0:
            return color, True
        best_color = color[:]
        best_total = total
        no_improve = 0
        max_no_improve = 3000
        it = 0
        # Precompute neighbor lists for faster access
        neigh_lists = [neigh[i] for i in range(m)]
        while time.time() < time_limit:
            if total == 0:
                return color, True
            # choose vertex: with prob 0.75 the most-conflicted, else random conflicting
            if total > 0:
                if rng.random() < 0.75:
                    v = -1
                    bv = -1
                    for x in range(m):
                        if conflict[x] > bv:
                            bv = conflict[x]
                            v = x
                else:
                    cand = [x for x in range(m) if conflict[x] > 0]
                    v = cand[rng.randrange(len(cand))]
            else:
                v = rng.randrange(m)
            old = color[v]
            # find best new colour (minimise conflict delta)
            best_c = old
            best_delta = 0
            best_delta_tie = 0
            for c in range(K):
                if c == old:
                    continue
                cnt_c = 0
                cnt_old = 0
                for u in neigh_lists[v]:
                    if color[u] == c:
                        cnt_c += 1
                    elif color[u] == old:
                        cnt_old += 1
                delta = cnt_c - cnt_old
                if delta < best_delta or (delta == best_delta and rng.random() < 0.3):
                    best_delta = delta
                    best_c = c
            # occasional random kick to escape local minima - increase frequency
            kick_prob = 0.15 + 0.05 * min(1.0, no_improve / 1000.0)
            if rng.random() < kick_prob:
                best_c = rng.randrange(K)
                cnt_c = 0
                cnt_old = 0
                for u in neigh_lists[v]:
                    if color[u] == best_c:
                        cnt_c += 1
                    elif color[u] == old:
                        cnt_old += 1
                best_delta = cnt_c - cnt_old
            if best_c == old:
                no_improve += 1
                continue
            # Apply the move
            for u in neigh_lists[v]:
                if color[u] == old:
                    conflict[u] -= 1
                    conflict[v] -= 1
                if color[u] == best_c:
                    conflict[u] += 1
                    conflict[v] += 1
            color[v] = best_c
            total += best_delta
            if total < best_total:
                best_total = total
                best_color = color[:]
                no_improve = 0
            else:
                no_improve += 1
            it += 1
            if no_improve > max_no_improve:
                # snap back to best and continue with a perturbation
                color = best_color[:]
                conflict, total = recompute_conflict(color, neigh, m)
                # Perturb a few random vertices to escape
                for _ in range(min(5, m // 10)):
                    v2 = rng.randrange(m)
                    color[v2] = rng.randrange(K)
                conflict, total = recompute_conflict(color, neigh, m)
                no_improve = 0
        return best_color, best_total == 0

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

    def solve_instance(n, d, seed, time_limit):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

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
        rng = Random(seed * 7919 + 42)

        # Best valid initial colouring: try several DSATUR variants, keep fewest colours
        best_color, best_K = dsatur_color(m, neigh, deg)
        for _ in range(8):
            col, K = dsatur_color(m, neigh, deg, rng=rng, rand_order=True)
            if K < best_K:
                best_color, best_K = col, K
        # Also try a simple greedy with different orderings
        for _ in range(4):
            col, K = dsatur_color(m, neigh, deg, rng=rng, rand_order=True)
            if K < best_K:
                best_color, best_K = col, K

        LB = greedy_clique(m, neigh)
        LB = max(LB, 1)

        # Descend K from best_K-1 down to LB, using min-conflicts local search
        current_color = best_color
        current_K = best_K
        for target_K in range(current_K - 1, LB - 1, -1):
            if time.time() > time_limit - 0.3:
                break
            # per-K time cap (a few seconds) but bounded by remaining
            remaining_time = time_limit - time.time()
            if remaining_time < 0.5:
                break
            cap = min(4.0, max(0.8, remaining_time * 0.6))
            k_deadline = time.time() + cap
            found = False
            restart = 0
            max_restarts = 15
            while time.time() < k_deadline and restart < max_restarts:
                if restart == 0:
                    init = current_color[:]
                elif restart % 3 == 1:
                    # Random initialization
                    init = [rng.randrange(target_K) for _ in range(m)]
                else:
                    # Perturbed current solution
                    init = current_color[:]
                    for _ in range(m // 3):
                        v = rng.randrange(m)
                        init[v] = rng.randrange(target_K)
                init = [c if c < target_K else rng.randrange(target_K) for c in init]
                # Give more time to earlier restarts
                time_per_restart = min(1.5, max(0.3, (k_deadline - time.time()) / max(1, max_restarts - restart)))
                res, ok = local_search(m, neigh, init, target_K, rng,
                                       min(k_deadline, time.time() + time_per_restart))
                if ok:
                    current_color = res
                    current_K = target_K
                    found = True
                    break
                restart += 1
            if not found:
                # Try one more aggressive attempt with longer time if we have budget
                if time.time() < time_limit - 1.0:
                    init = [rng.randrange(target_K) for _ in range(m)]
                    res, ok = local_search(m, neigh, init, target_K, rng, time_limit - 0.5)
                    if ok:
                        current_color = res
                        current_K = target_K
                else:
                    break
        return compress(current_color)

    def fallback():
        out = []
        for spec in specs:
            n, d, seed = spec
            edges = instance(n, d, seed)
            m = len(edges)
            if m == 0:
                out.append([])
                continue
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
                        neigh[i].append(j); neigh[j].append(i)
                    elif (c in adj_orig[a] or dd in adj_orig[a] or
                          c in adj_orig[b] or dd in adj_orig[b]):
                        neigh[i].append(j); neigh[j].append(i)
            col, _ = dsatur_color(m, neigh, [len(neigh[i]) for i in range(m)])
            out.append(col)
        return out

    try:
        start = time.time()
        master_deadline = start + 117.0
        # allocate time by difficulty weight d^2, with more weight to later (harder) instances
        weights = [d * d * (1.0 + idx * 0.1) for idx, (_, d, _) in enumerate(specs)]
        total_w = sum(weights)
        out = []
        cursor = start
        for idx, (n, d, seed) in enumerate(specs):
            frac = weights[idx] / total_w
            remaining = master_deadline - cursor
            if remaining < 2.0:
                # Not enough time, use quick fallback for remaining
                for jdx in range(idx, len(specs)):
                    nj, dj, sj = specs[jdx]
                    out.append(fallback_instance(nj, dj, sj))
                return out
            budget = remaining * frac
            budget = max(budget, 4.0)
            inst_deadline = min(cursor + budget, master_deadline)
            out.append(solve_instance(n, d, seed, inst_deadline))
            cursor = max(cursor, time.time())
        return out
    except Exception:
        return fallback()

    def fallback_instance(n, d, seed):
        """Quick fallback for a single instance."""
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
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
                    neigh[i].append(j); neigh[j].append(i)
                elif (c in adj_orig[a] or dd in adj_orig[a] or
                      c in adj_orig[b] or dd in adj_orig[b]):
                    neigh[i].append(j); neigh[j].append(i)
        col, _ = dsatur_color(m, neigh, [len(neigh[i]) for i in range(m)])
        return col

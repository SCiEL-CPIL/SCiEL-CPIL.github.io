import time
import random
import math
from itertools import combinations


def construct():
    start = time.time()
    deadline = start + 80.0

    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    m_total = len(all_edges)

    def make_scoring(values):
        sv = values[:]
        cnt_sum = {}
        cnt_prod = {}
        distinct_sum = 0
        distinct_prod = 0

        def add(x, cnt, which):
            nonlocal distinct_sum, distinct_prod
            c = cnt.get(x, 0)
            if c == 0:
                if which == 's':
                    distinct_sum += 1
                else:
                    distinct_prod += 1
            cnt[x] = c + 1

        def rem(x, cnt, which):
            nonlocal distinct_sum, distinct_prod
            c = cnt.get(x, 0)
            if c == 1:
                del cnt[x]
                if which == 's':
                    distinct_sum -= 1
                else:
                    distinct_prod -= 1
            else:
                cnt[x] = c - 1

        def swap(out, inn):
            u, v = out
            s0 = sv[u] + sv[v]
            p0 = sv[u] * sv[v]
            rem(s0, cnt_sum, 's')
            rem(p0, cnt_prod, 'p')
            u, v = inn
            s1 = sv[u] + sv[v]
            p1 = sv[u] * sv[v]
            add(s1, cnt_sum, 's')
            add(p1, cnt_prod, 'p')

        def add_edge(e):
            u, v = e
            add(sv[u] + sv[v], cnt_sum, 's')
            add(sv[u] * sv[v], cnt_prod, 'p')

        def remove_edge(e):
            u, v = e
            rem(sv[u] + sv[v], cnt_sum, 's')
            rem(sv[u] * sv[v], cnt_prod, 'p')

        def raw():
            return max(distinct_sum, distinct_prod)

        def init_from(edges):
            for e in edges:
                add_edge(e)

        return init_from, swap, add_edge, remove_edge, raw, (cnt_sum, cnt_prod)

    def full_score(values, edge_list):
        s = set()
        p = set()
        for u, v in edge_list:
            s.add(values[u] + values[v])
            p.add(values[u] * values[v])
        return max(len(s), len(p))

    # Value generation strategies
    def generate_value_sets():
        sets = []

        # Consecutive small values
        for start_v in range(1, 12):
            for span in range(14, 30):
                pool = list(range(start_v, start_v + span))
                if len(pool) >= n:
                    sets.append(random.sample(pool, n))

        # Arithmetic progressions
        for start_v in range(1, 30):
            for step in [1, 2, 3, 4, 5, 6, 7, 8]:
                pool = [start_v + i * step for i in range(40) if start_v + i * step <= 200]
                if len(pool) >= n:
                    sets.append(random.sample(pool, n))

        # Multiples of small numbers
        for base in [2, 3, 4, 5, 6, 8, 10, 12]:
            pool = [i * base for i in range(1, 30)]
            if len(pool) >= n:
                sets.append(random.sample(pool, n))

        # Clusters around centers
        for center in range(5, 100, 5):
            for spread in [3, 5, 8, 12, 15]:
                seen = set()
                vals = []
                for _ in range(50):
                    v = center + random.randint(-spread, spread)
                    if 1 <= v <= 10000 and v not in seen:
                        seen.add(v)
                        vals.append(v)
                    if len(vals) == n:
                        break
                if len(vals) == n:
                    sets.append(vals)

        # Powers and structured
        base_sets = [
            [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 35,
             36, 40, 45, 48, 49, 50, 54, 56, 60, 63, 64, 70, 72, 75, 80, 81, 84, 90],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
             42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76],
        ]
        for bs in base_sets:
            if len(bs) >= n:
                sets.append(random.sample(bs, n))

        # Random small values
        for _ in range(30):
            seen = set()
            vals = []
            while len(vals) < n:
                v = random.randint(1, 60)
                if v not in seen:
                    seen.add(v)
                    vals.append(v)
            sets.append(vals)

        random.shuffle(sets)
        return sets

    def greedy_edges(values):
        chosen = []
        chosen_set = set()
        cnt_sum = {}
        cnt_prod = {}
        distinct_sum = 0
        distinct_prod = 0

        # Sort edges by sum (prioritize small sums for collisions)
        order = sorted(all_edges, key=lambda e: values[e[0]] + values[e[1]])

        for e in order:
            if len(chosen) == m:
                break
            u, v = e
            s = values[u] + values[v]
            p = values[u] * values[v]

            cnt_sum[s] = cnt_sum.get(s, 0) + 1
            cnt_prod[p] = cnt_prod.get(p, 0) + 1
            if cnt_sum[s] == 1:
                distinct_sum += 1
            if cnt_prod[p] == 1:
                distinct_prod += 1

            chosen.append(e)
            chosen_set.add(e)

        return chosen, chosen_set

    def polish_edges(values, chosen, chosen_set, steps, deadline):
        init_from, swap, add_edge, remove_edge, raw, _ = make_scoring(values)
        init_from(chosen)
        cur = raw()
        best = cur
        no_improve = 0

        for _ in range(steps):
            if time.time() > deadline:
                break

            i = random.randrange(m)
            e_out = chosen[i]

            best_e_in = None
            best_r = float('inf')
            best_swap_data = None

            for _ in range(8):
                e_in = random.choice(all_edges)
                if e_in in chosen_set:
                    continue

                swap(e_out, e_in)
                r = raw()

                if r < best_r:
                    best_r = r
                    best_e_in = e_in
                    best_swap_data = (e_out, e_in)

                swap(e_in, e_out)

            if best_e_in is None:
                no_improve += 1
                if no_improve > 20000:
                    break
                continue

            e_out, e_in = best_swap_data
            swap(e_out, e_in)
            r = raw()

            if r <= cur:
                cur = r
                chosen[i] = e_in
                chosen_set.discard(e_out)
                chosen_set.add(e_in)
                if r < best:
                    best = r
                no_improve = 0
            else:
                swap(e_in, e_out)
                no_improve += 1

        return chosen, chosen_set, best

    def joint_search(vals0, edges0, steps, deadline):
        vals = list(vals0)
        edges = list(edges0)
        edges_set = set(edges0)
        cur = full_score(vals, edges)
        best = cur
        best_vals = list(vals)
        best_edges = list(edges)
        init_from, swap, add_edge, remove_edge, raw, _ = make_scoring(vals)
        init_from(edges)
        no_improve = 0

        for _ in range(steps):
            if time.time() > deadline:
                break

            if random.random() < 0.55:
                # Edge move
                i = random.randrange(m)
                e_out = edges[i]

                best_e_in = None
                best_r = float('inf')
                best_swap_data = None

                for _ in range(5):
                    e_in = random.choice(all_edges)
                    if e_in in edges_set:
                        continue

                    swap(e_out, e_in)
                    r = raw()

                    if r < best_r:
                        best_r = r
                        best_e_in = e_in
                        best_swap_data = (e_out, e_in)

                    swap(e_in, e_out)

                if best_e_in is None:
                    no_improve += 1
                    continue

                e_out, e_in = best_swap_data
                swap(e_out, e_in)
                r = raw()

                if r <= cur:
                    cur = r
                    edges[i] = e_in
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    if r < best:
                        best = r
                        best_vals = list(vals)
                        best_edges = list(edges)
                    no_improve = 0
                else:
                    swap(e_in, e_out)
                    no_improve += 1
            else:
                # Value move
                idx = random.randrange(n)
                old = vals[idx]

                candidates = []
                for _ in range(4):
                    candidates.append(random.randint(1, 40))
                for _ in range(2):
                    candidates.append(random.randint(10, 80))
                for _ in range(3):
                    other_idx = random.randrange(n)
                    if other_idx != idx:
                        d = random.randint(-8, 8)
                        nv = vals[other_idx] + d
                        if 1 <= nv <= 10000:
                            candidates.append(nv)
                        nv2 = vals[other_idx] * random.choice([2, 3, 5])
                        if nv2 <= 10000:
                            candidates.append(nv2)

                best_nv = None
                best_r = float('inf')

                for nv in candidates:
                    if nv < 1 or nv > 10000 or nv == old or nv in vals:
                        continue

                    vals[idx] = nv
                    r = full_score(vals, edges)

                    if r < best_r:
                        best_r = r
                        best_nv = nv

                    vals[idx] = old

                if best_nv is None:
                    no_improve += 1
                    continue

                vals[idx] = best_nv
                r = best_r

                if r <= cur:
                    cur = r
                    init_from, swap, add_edge, remove_edge, raw, _ = make_scoring(vals)
                    init_from(edges)
                    if r < best:
                        best = r
                        best_vals = list(vals)
                        best_edges = list(edges)
                    no_improve = 0
                else:
                    vals[idx] = old
                    no_improve += 1

        return best, best_vals, best_edges

    # Driver
    best_score = float('inf')
    best_vals = None
    best_edges = None

    def consider(sc, vals, edges):
        nonlocal best_score, best_vals, best_edges
        if sc < best_score:
            best_score = sc
            best_vals = list(vals)
            best_edges = list(edges)

    # Phase 1: Try multiple value sets with greedy edges + edge polish
    value_sets = generate_value_sets()
    random.shuffle(value_sets)

    for vals in value_sets[:40]:
        if time.time() > deadline - 15.0:
            break
        chosen, chosen_set = greedy_edges(vals)
        sc = full_score(vals, chosen)
        consider(sc, vals, chosen)

        steps = max(3000, int((deadline - 12.0 - time.time()) / 0.00015))
        polished, pset, psc = polish_edges(vals, chosen[:], chosen_set.copy(),
                                           steps, deadline - 10.0)
        consider(psc, vals, polished)

    # Phase 2: Joint search from best and diverse starts
    starts = []
    if best_vals is not None:
        starts.append((list(best_vals), list(best_edges)))

    for _ in range(6):
        vals = value_sets[random.randrange(len(value_sets))]
        edges = random.sample(all_edges, m)
        starts.append((vals, edges))

    for vals in value_sets[:3]:
        chosen, _ = greedy_edges(vals)
        starts.append((list(vals), list(chosen)))

    remaining = deadline - 8.0 - time.time()
    if remaining < 3.0:
        remaining = 3.0
    per = max(4000, int(remaining / max(1, len(starts)) / 0.0002))

    for vals0, edges0 in starts:
        if time.time() > deadline - 7.0:
            break
        sc, vals, edges = joint_search(vals0, edges0, per, deadline - 6.0)
        consider(sc, vals, edges)

    # Phase 3: Final edge polish on best
    if time.time() < deadline - 4.0 and best_vals is not None:
        chosen, chosen_set = list(best_edges), set(best_edges)
        polished, pset, psc = polish_edges(best_vals, chosen, chosen_set,
                                           80000, deadline - 3.0)
        consider(psc, best_vals, polished)

    # Phase 4: Final value polish on best edges
    if time.time() < deadline - 2.0 and best_edges is not None:
        vals = list(best_vals)
        edges = list(best_edges)
        cur = full_score(vals, edges)
        best = cur
        it = 0
        no_improve = 0

        while it < 20000 and time.time() < deadline - 1.5:
            it += 1
            idx = random.randrange(n)
            old = vals[idx]

            candidates = []
            for _ in range(5):
                candidates.append(random.randint(1, 50))
            for _ in range(3):
                other_idx = random.randrange(n)
                if other_idx != idx:
                    d = random.randint(-6, 6)
                    nv = vals[other_idx] + d
                    if 1 <= nv <= 10000:
                        candidates.append(nv)

            best_nv = None
            best_r = float('inf')

            for nv in candidates:
                if nv < 1 or nv > 10000 or nv == old or nv in vals:
                    continue

                vals[idx] = nv
                r = full_score(vals, edges)

                if r < best_r:
                    best_r = r
                    best_nv = nv

                vals[idx] = old

            if best_nv is None:
                no_improve += 1
                if no_improve > 3000:
                    break
                continue

            vals[idx] = best_nv
            r = best_r

            if r < cur:
                cur = r
                if r < best:
                    best = r
                    best_vals = list(vals)
                    best_score = best
                no_improve = 0
            else:
                vals[idx] = old
                no_improve += 1

    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])
        best_score = full_score(best_vals, best_edges)

    # Ensure validity
    seen_vals = set()
    clean_vals = []
    for v in best_vals:
        if v not in seen_vals and 1 <= v <= 10000:
            seen_vals.add(v)
            clean_vals.append(v)
    while len(clean_vals) < n:
        v = random.randint(1, 10000)
        if v not in seen_vals:
            seen_vals.add(v)
            clean_vals.append(v)
    best_vals = clean_vals[:n]

    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    seen = set()
    clean = []
    for e in final_edges:
        a, b = e
        if a == b or a < 0 or b < 0 or a >= n or b >= n:
            continue
        t = tuple(sorted((a, b)))
        if t in seen:
            continue
        seen.add(t)
        clean.append([a, b])
    if len(clean) < m:
        for e in all_edges:
            t = tuple(sorted(e))
            if t not in seen:
                seen.add(t)
                clean.append([e[0], e[1]])
                if len(clean) == m:
                    break
    final_edges = clean[:m]

    return {'values': final_values, 'edges': final_edges}

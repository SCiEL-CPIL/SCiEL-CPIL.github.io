def construct():
    import time
    from random import Random
    from collections import defaultdict

    t0 = time.time()
    DEADLINE = t0 + 114.0

    def gen_instance(n, d, seed):
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

    def build_conflict(edges):
        m = len(edges)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        conf = [set() for _ in range(m)]
        v_to_e = {}
        for i, (u, v) in enumerate(edges):
            v_to_e.setdefault(u, []).append(i)
            v_to_e.setdefault(v, []).append(i)
        for i in range(m):
            u, v = edges[i]
            near = {u, v} | (adj.get(u, set()) | adj.get(v, set()))
            for x in near:
                for j in v_to_e.get(x, []):
                    if j != i and j > i:
                        conf[i].add(j)
                        conf[j].add(i)
        conf_list = [tuple(sorted(c)) for c in conf]
        return conf_list

    def dsatur_coloring(conf, m, deadline):
        col = [-1] * m
        forbidden = [set() for _ in range(m)]
        remaining = set(range(m))
        while remaining and time.time() < deadline:
            best_edge = None
            best_count = -1
            for e in remaining:
                cnt = len(forbidden[e])
                if cnt > best_count:
                    best_count = cnt
                    best_edge = e
            if best_edge is None:
                best_edge = next(iter(remaining))
            c = 0
            while c in forbidden[best_edge]:
                c += 1
            col[best_edge] = c
            remaining.discard(best_edge)
            for j in conf[best_edge]:
                if j in remaining:
                    forbidden[j].add(c)
        return col

    def k_of(col):
        if not col:
            return 0
        return max(col) + 1

    def reindex(col):
        present = sorted(set(col))
        if not present:
            return col
        remap = {c: i for i, c in enumerate(present)}
        for i in range(len(col)):
            col[i] = remap[col[i]]
        return col

    def is_valid_move(col, e, new_c, conf):
        for j in conf[e]:
            if col[j] == new_c:
                return False
        return True

    def local_improve(col, conf, m, deadline, rng):
        col = list(col)
        improved = True
        guard = 0
        while improved and time.time() < deadline and guard < 200:
            improved = False
            guard += 1
            cls = defaultdict(set)
            for i, c in enumerate(col):
                cls[c].add(i)

            # Merge pass
            while time.time() < deadline:
                merged = False
                cols = sorted(cls.keys())
                for idx, c1 in enumerate(cols):
                    if c1 not in cls:
                        continue
                    s1 = cls[c1]
                    for c2 in cols[idx + 1:]:
                        if c2 not in cls:
                            continue
                        s2 = cls[c2]
                        ok = True
                        for e in s2:
                            for j in conf[e]:
                                if j in s1:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            for e in s2:
                                col[e] = c1
                            s1.update(s2)
                            del cls[c2]
                            merged = True
                            improved = True
                            break
                    if merged:
                        break
                if not merged:
                    break

            # Single-edge moves to lower colours
            if not improved and time.time() < deadline:
                cls = defaultdict(set)
                for i, c in enumerate(col):
                    cls[c].add(i)
                cols = sorted(cls.keys())
                if len(cols) >= 2:
                    moved = False
                    for _ in range(8000):
                        e = rng.randrange(m)
                        c = col[e]
                        for new_c in range(c):
                            if is_valid_move(col, e, new_c, conf):
                                col[e] = new_c
                                moved = True
                                break
                        if moved:
                            break
                    if moved:
                        improved = True

            # 2-swap
            if not improved and time.time() < deadline:
                cls = defaultdict(set)
                for i, c in enumerate(col):
                    cls[c].add(i)
                cols = sorted(cls.keys())
                if len(cols) >= 2:
                    swapped = False
                    for _ in range(4000):
                        e1 = rng.randrange(m)
                        e2 = rng.randrange(m)
                        if e1 == e2:
                            continue
                        c1, c2 = col[e1], col[e2]
                        if c1 == c2:
                            continue
                        valid = True
                        for j in conf[e1]:
                            if j != e2 and col[j] == c2:
                                valid = False
                                break
                        if valid:
                            for j in conf[e2]:
                                if j != e1 and col[j] == c1:
                                    valid = False
                                    break
                        if valid:
                            col[e1], col[e2] = c2, c1
                            swapped = True
                            break
                    if swapped:
                        improved = True

            # 3-swap
            if not improved and time.time() < deadline:
                cls = defaultdict(set)
                for i, c in enumerate(col):
                    cls[c].add(i)
                cols = sorted(cls.keys())
                if len(cols) >= 3:
                    tri_swapped = False
                    for _ in range(2000):
                        e1 = rng.randrange(m)
                        e2 = rng.randrange(m)
                        e3 = rng.randrange(m)
                        if e1 == e2 or e2 == e3 or e1 == e3:
                            continue
                        c1, c2, c3 = col[e1], col[e2], col[e3]
                        if len({c1, c2, c3}) < 3:
                            continue
                        valid = True
                        for j in conf[e1]:
                            if j not in (e2, e3) and col[j] == c2:
                                valid = False
                                break
                        if valid:
                            for j in conf[e2]:
                                if j not in (e1, e3) and col[j] == c3:
                                    valid = False
                                    break
                        if valid:
                            for j in conf[e3]:
                                if j not in (e1, e2) and col[j] == c1:
                                    valid = False
                                    break
                        if valid:
                            col[e1], col[e2], col[e3] = c2, c3, c1
                            tri_swapped = True
                            break
                    if tri_swapped:
                        improved = True

        return reindex(col)

    def simulated_annealing(col, conf, m, deadline, rng, iterations=300000):
        col = list(col)
        best_col = col[:]
        best_k = k_of(col)
        T = 1.2
        cool = 0.99998
        max_c = best_k

        count_arr = [0] * (max_c + 2)
        for c in col:
            if c < len(count_arr):
                count_arr[c] += 1
            else:
                count_arr.extend([0] * (c + 1 - len(count_arr)))
                count_arr[c] += 1

        for it in range(iterations):
            if time.time() > deadline:
                break
            e = rng.randrange(m)
            c_old = col[e]

            r = rng.random()
            if r < 0.45:
                new_c = rng.randrange(c_old) if c_old > 0 else 0
            elif r < 0.75:
                new_c = rng.randrange(max(1, best_k))
            else:
                new_c = rng.randrange(max_c + 2)

            if new_c == c_old:
                continue

            valid = True
            for j in conf[e]:
                if col[j] == new_c:
                    valid = False
                    break

            if valid:
                if new_c >= len(count_arr):
                    count_arr.extend([0] * (new_c + 1 - len(count_arr)))
                count_arr[c_old] -= 1
                count_arr[new_c] += 1
                col[e] = new_c

                # Compute new k
                new_k = 0
                for i in range(len(count_arr) - 1, -1, -1):
                    if count_arr[i] > 0:
                        new_k = i + 1
                        break

                if new_k < best_k:
                    best_col = col[:]
                    best_k = new_k
                    max_c = best_k
                    T = max(0.01, T * 0.9)
                else:
                    delta = new_k - best_k
                    if delta == 0:
                        if rng.random() >= 0.5:
                            count_arr[new_c] -= 1
                            count_arr[c_old] += 1
                            col[e] = c_old
                    else:
                        accept_prob = 2.718281828 ** (-delta / max(T, 0.01))
                        if rng.random() >= accept_prob:
                            count_arr[new_c] -= 1
                            count_arr[c_old] += 1
                            col[e] = c_old

                T = max(0.01, T * cool)

        return best_col

    def targeted_elimination(col, conf, m, deadline, rng):
        """Try to eliminate the highest colour by moving its edges to lower colours."""
        col = list(col)
        cur_k = k_of(col)
        if cur_k <= 1:
            return col

        # Edges with the highest colour
        top_c = cur_k - 1
        top_edges = [i for i in range(m) if col[i] == top_c]

        for attempt in range(200):
            if time.time() > deadline:
                break
            # Try to move each top-edge to a lower colour
            for e in top_edges[:]:
                if col[e] != top_c:
                    continue
                found = False
                for new_c in range(top_c):
                    if is_valid_move(col, e, new_c, conf):
                        col[e] = new_c
                        found = True
                        break
                if not found:
                    # Try a 2-swap: find an edge f with colour c_f such that
                    # e can take c_f and f can take top_c
                    for f in range(m):
                        if f == e or col[f] == top_c:
                            continue
                        c_f = col[f]
                        # Check e can take c_f
                        ok_e = True
                        for j in conf[e]:
                            if j != f and col[j] == c_f:
                                ok_e = False
                                break
                        if not ok_e:
                            continue
                        # Check f can take top_c
                        ok_f = True
                        for j in conf[f]:
                            if j != e and col[j] == top_c:
                                ok_f = False
                                break
                        if ok_f:
                            col[e] = c_f
                            col[f] = top_c
                            found = True
                            break
                if found:
                    top_edges = [i for i in range(m) if col[i] == top_c]
                    if not top_edges:
                        break
            if not top_edges:
                break

        return reindex(col)

    def solve_one(spec, time_budget):
        n, d, seed = spec
        edges = gen_instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        conf = build_conflict(edges)

        best_colors = None
        best_num_colors = float('inf')

        # Phase 1: Multiple DSATUR + local improvement
        for trial in range(8):
            if time.time() > t0 + time_budget:
                break
            rng = Random(seed + 1000 * trial)
            col = dsatur_coloring(conf, m, t0 + time_budget)
            col = local_improve(col, conf, m, t0 + time_budget, rng)
            k = k_of(col)
            if k < best_num_colors:
                best_num_colors = k
                best_colors = col[:]

        # Phase 2: SA from best
        if best_colors is not None and time.time() < t0 + time_budget:
            for sa_trial in range(4):
                if time.time() > t0 + time_budget:
                    break
                rng = Random(seed + 99999 + 1000 * sa_trial)
                sa_col = simulated_annealing(best_colors, conf, m, t0 + time_budget, rng, iterations=200000)
                sa_k = k_of(sa_col)
                if sa_k < best_num_colors:
                    best_num_colors = sa_k
                    best_colors = sa_col[:]
                sa_col = local_improve(sa_col, conf, m, t0 + time_budget, rng)
                sa_k = k_of(sa_col)
                if sa_k < best_num_colors:
                    best_num_colors = sa_k
                    best_colors = sa_col[:]

        # Phase 3: Targeted elimination
        if best_colors is not None and time.time() < t0 + time_budget:
            for elim_pass in range(3):
                if time.time() > t0 + time_budget:
                    break
                rng = Random(seed + 55555 + 100 * elim_pass)
                elim_col = targeted_elimination(best_colors, conf, m, t0 + time_budget, rng)
                ek = k_of(elim_col)
                if ek < best_num_colors:
                    best_num_colors = ek
                    best_colors = elim_col[:]
                elim_col = local_improve(elim_col, conf, m, t0 + time_budget, rng)
                ek = k_of(elim_col)
                if ek < best_num_colors:
                    best_num_colors = ek
                    best_colors = elim_col[:]

        # Phase 4: Final SA polish
        if best_colors is not None and time.time() < t0 + time_budget:
            rng = Random(seed + 123456)
            final_col = simulated_annealing(best_colors, conf, m, t0 + time_budget, rng, iterations=100000)
            fk = k_of(final_col)
            if fk < best_num_colors:
                best_num_colors = fk
                best_colors = final_col[:]
                if time.time() < t0 + time_budget:
                    final_col = local_improve(best_colors, conf, m, t0 + time_budget, rng)
                    fk = k_of(final_col)
                    if fk < best_num_colors:
                        best_num_colors = fk
                        best_colors = final_col

        if best_colors is None:
            best_colors = dsatur_coloring(conf, m, t0 + time_budget)

        return best_colors

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    total_time = 110.0
    weights = [n * d for n, d, s in specs]
    total_w = sum(weights)
    budgets = [total_time * w / total_w for w in weights]

    out = []
    for i, spec in enumerate(specs):
        if time.time() > t0 + DEADLINE - 5:
            edges = gen_instance(*spec)
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
        else:
            out.append(solve_one(spec, budgets[i]))

    return out

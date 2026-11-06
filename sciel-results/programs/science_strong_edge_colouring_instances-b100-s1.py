def construct():
    import time
    from random import Random
    from collections import defaultdict
    import heapq

    t0 = time.time()
    DEADLINE = t0 + 115.0  # safety margin under the 150 s kill

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
        # Precompute conflict lists as sorted tuples for faster iteration
        conf_list = [tuple(sorted(c)) for c in conf]
        return conf_list

    def dsatur_coloring(conf, m, deadline):
        """DSATUR-style coloring: prioritize edges with most colored conflicting neighbors."""
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
        return max(col) + 1 if col else 0

    def k_of_fast(col, m):
        """Track max colour incrementally."""
        if m == 0:
            return 0
        # Use a boolean array for O(1) max query
        present = bytearray(max(col) + 1)
        for c in col:
            present[c] = 1
        # Find highest set bit
        for i in range(len(present) - 1, -1, -1):
            if present[i]:
                return i + 1
        return 0

    def build_classes(col):
        cls = defaultdict(set)
        for i, c in enumerate(col):
            cls[c].add(i)
        return cls

    def reindex(col):
        present = sorted(set(col))
        if not present:
            return col
        remap = {c: i for i, c in enumerate(present)}
        for i in range(len(col)):
            col[i] = remap[col[i]]
        return col

    def is_valid_move(col, e, new_c, conf):
        """Check if edge e can be assigned colour new_c."""
        for j in conf[e]:
            if col[j] == new_c:
                return False
        return True

    def local_improve(col, conf, m, deadline, rng):
        col = list(col)
        improved = True
        guard = 0
        while improved and time.time() < deadline and guard < 1000:
            improved = False
            guard += 1
            cls = build_classes(col)
            
            # --- merge pass: fold one whole class into another when compatible ---
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

            # --- single-edge moves: try to move an edge to any valid lower colour ---
            if not improved and time.time() < deadline:
                cls = build_classes(col)
                cols = sorted(cls.keys())
                if len(cols) >= 2:
                    moved = False
                    for _ in range(10000):
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

            # --- 2-swap moves ---
            if not improved and time.time() < deadline:
                cls = build_classes(col)
                cols = sorted(cls.keys())
                if len(cols) >= 2:
                    swapped = False
                    for _ in range(5000):
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

            # --- 3-swap moves: try to find a cycle of 3 edges that can be rotated ---
            if not improved and time.time() < deadline:
                cls = build_classes(col)
                cols = sorted(cls.keys())
                if len(cols) >= 3:
                    tri_swapped = False
                    for _ in range(3000):
                        e1 = rng.randrange(m)
                        e2 = rng.randrange(m)
                        e3 = rng.randrange(m)
                        if e1 == e2 or e2 == e3 or e1 == e3:
                            continue
                        c1, c2, c3 = col[e1], col[e2], col[e3]
                        if len({c1, c2, c3}) < 3:
                            continue
                        # Check if e1->c2, e2->c3, e3->c1 is valid
                        valid = True
                        for j in conf[e1]:
                            if j != e2 and j != e3 and col[j] == c2:
                                valid = False
                                break
                        if valid:
                            for j in conf[e2]:
                                if j != e1 and j != e3 and col[j] == c3:
                                    valid = False
                                    break
                        if valid:
                            for j in conf[e3]:
                                if j != e1 and j != e2 and col[j] == c1:
                                    valid = False
                                    break
                        if valid:
                            col[e1], col[e2], col[e3] = c2, c3, c1
                            tri_swapped = True
                            break
                    if tri_swapped:
                        improved = True

            # --- 4-swap moves: try to find a cycle of 4 edges that can be rotated ---
            if not improved and time.time() < deadline:
                cls = build_classes(col)
                cols = sorted(cls.keys())
                if len(cols) >= 4:
                    quad_swapped = False
                    for _ in range(1000):
                        e1 = rng.randrange(m)
                        e2 = rng.randrange(m)
                        e3 = rng.randrange(m)
                        e4 = rng.randrange(m)
                        if len({e1, e2, e3, e4}) < 4:
                            continue
                        c1, c2, c3, c4 = col[e1], col[e2], col[e3], col[e4]
                        if len({c1, c2, c3, c4}) < 4:
                            continue
                        # Check if e1->c2, e2->c3, e3->c4, e4->c1 is valid
                        valid = True
                        for j in conf[e1]:
                            if j not in (e2, e3, e4) and col[j] == c2:
                                valid = False
                                break
                        if valid:
                            for j in conf[e2]:
                                if j not in (e1, e3, e4) and col[j] == c3:
                                    valid = False
                                    break
                        if valid:
                            for j in conf[e3]:
                                if j not in (e1, e2, e4) and col[j] == c4:
                                    valid = False
                                    break
                        if valid:
                            for j in conf[e4]:
                                if j not in (e1, e2, e3) and col[j] == c1:
                                    valid = False
                                    break
                        if valid:
                            col[e1], col[e2], col[e3], col[e4] = c2, c3, c4, c1
                            quad_swapped = True
                            break
                    if quad_swapped:
                        improved = True

        return reindex(col)

    def simulated_annealing(col, conf, m, deadline, rng, iterations=200000):
        col = list(col)
        best_col = col[:]
        best_k = k_of(col)
        T = 1.5
        cool = 0.99997
        n_accept = 0
        n_total = 0
        # Track colour counts for fast k computation
        max_c = best_k
        # Maintain count array for O(1) max query
        count_arr = [0] * (max_c + 1)
        for c in col:
            count_arr[c] += 1
        
        def get_k_from_count():
            for i in range(len(count_arr) - 1, -1, -1):
                if count_arr[i] > 0:
                    return i + 1
            return 0
        
        for it in range(iterations):
            if time.time() > deadline:
                break
            n_total += 1
            e = rng.randrange(m)
            c_old = col[e]
            # Pick a random new colour (biased towards lower colours)
            r = rng.random()
            if r < 0.4:
                new_c = rng.randrange(c_old) if c_old > 0 else 0
            elif r < 0.7:
                new_c = rng.randrange(max(1, c_old + 1))
            else:
                new_c = rng.randrange(max_c + 3)
            
            if new_c == c_old:
                continue
            
            # Check validity
            valid = True
            for j in conf[e]:
                if col[j] == new_c:
                    valid = False
                    break
            
            if valid:
                # Update count array
                count_arr[c_old] -= 1
                if new_c >= len(count_arr):
                    count_arr.extend([0] * (new_c + 1 - len(count_arr)))
                count_arr[new_c] += 1
                
                col[e] = new_c
                new_k = get_k_from_count()
                
                if new_k < best_k:
                    best_col = col[:]
                    best_k = new_k
                    max_c = best_k
                    n_accept += 1
                    T = max(0.01, T * 0.95)
                else:
                    delta = new_k - best_k
                    if delta == 0:
                        if rng.random() < 0.4:
                            n_accept += 1
                        else:
                            # Revert
                            count_arr[new_c] -= 1
                            count_arr[c_old] += 1
                            col[e] = c_old
                    else:
                        if rng.random() < (2.718281828 ** (-delta / T)):
                            n_accept += 1
                        else:
                            # Revert
                            count_arr[new_c] -= 1
                            count_arr[c_old] += 1
                            col[e] = c_old
                T = max(0.01, T * cool)
            else:
                pass
        
        return best_col

    def solve_one(spec, time_budget):
        n, d, seed = spec
        edges = gen_instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        conf = build_conflict(edges)
        
        best_colors = None
        best_num_colors = float('inf')
        
        # Phase 1: DSATUR + local improvement (fast, good baseline)
        for trial in range(10):
            if time.time() > t0 + time_budget:
                break
            rng = Random(seed + 1000 * trial)
            col = dsatur_coloring(conf, m, t0 + time_budget)
            col = local_improve(col, conf, m, t0 + time_budget, rng)
            k = k_of(col)
            if k < best_num_colors:
                best_num_colors = k
                best_colors = col[:]
        
        # Phase 2: Simulated annealing from best solution with multiple restarts
        if best_colors is not None and time.time() < t0 + time_budget:
            # Run SA multiple times with different seeds
            for sa_trial in range(5):
                if time.time() > t0 + time_budget:
                    break
                rng = Random(seed + 99999 + 1000 * sa_trial)
                sa_col = simulated_annealing(best_colors, conf, m, t0 + time_budget, rng, iterations=120000)
                sa_k = k_of(sa_col)
                if sa_k < best_num_colors:
                    best_num_colors = sa_k
                    best_colors = sa_col[:]
                # Local improve the SA result
                sa_col = local_improve(sa_col, conf, m, t0 + time_budget, rng)
                sa_k = k_of(sa_col)
                if sa_k < best_num_colors:
                    best_num_colors = sa_k
                    best_colors = sa_col[:]
        
        # Phase 3: Final polish - try more aggressive local search with multiple passes
        if best_colors is not None and time.time() < t0 + time_budget:
            for polish_pass in range(3):
                if time.time() > t0 + time_budget:
                    break
                rng = Random(seed + 77777 + 100 * polish_pass)
                polished = local_improve(best_colors, conf, m, t0 + time_budget, rng)
                pk = k_of(polished)
                if pk < best_num_colors:
                    best_num_colors = pk
                    best_colors = polished[:]
        
        # Phase 4: Final SA polish with very low temperature
        if best_colors is not None and time.time() < t0 + time_budget:
            rng = Random(seed + 123456)
            final_col = simulated_annealing(best_colors, conf, m, t0 + time_budget, rng, iterations=50000)
            fk = k_of(final_col)
            if fk < best_num_colors:
                best_num_colors = fk
                best_colors = final_col[:]
                # One more local improve
                if time.time() < t0 + time_budget:
                    final_col = local_improve(best_colors, conf, m, t0 + time_budget, rng)
                    fk = k_of(final_col)
                    if fk < best_num_colors:
                        best_num_colors = fk
                        best_colors = final_col
        
        if best_colors is None:
            # Fallback: greedy
            best_colors = dsatur_coloring(conf, m, t0 + time_budget)
        
        return best_colors

    # Allocate time budget to each instance
    # Larger instances get more time
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    total_time = 110.0  # leave 5s margin
    # Allocate proportional to n*d (roughly proportional to problem size)
    weights = [n * d for n, d, s in specs]
    total_w = sum(weights)
    budgets = [total_time * w / total_w for w in weights]
    
    out = []
    for i, spec in enumerate(specs):
        if time.time() > t0 + DEADLINE - 5:
            # Emergency fallback: greedy first-fit
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

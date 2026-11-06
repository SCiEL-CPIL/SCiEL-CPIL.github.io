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
        """DSATUR-style coloring with MRV and forward checking."""
        col = [-1] * m
        forbidden = [set() for _ in range(m)]
        remaining = list(range(m))
        remaining_set = set(range(m))
        
        while remaining and time.time() < deadline:
            # MRV: pick edge with fewest available colors (most forbidden)
            best_edge = None
            best_mrv = -1
            for e in remaining:
                cnt = len(forbidden[e])
                if cnt > best_mrv:
                    best_mrv = cnt
                    best_edge = e
            
            if best_edge is None:
                best_edge = remaining[0]
            
            c = 0
            while c in forbidden[best_edge]:
                c += 1
            
            col[best_edge] = c
            remaining_set.discard(best_edge)
            remaining.remove(best_edge)
            
            for j in conf[best_edge]:
                if j in remaining_set:
                    forbidden[j].add(c)
        
        return col

    def k_of(col):
        return max(col) + 1 if col else 0

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
        # Track max color for O(1) k_of
        max_c = max(col) if col else -1
        
        while improved and time.time() < deadline and guard < 500:
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
                    for _ in range(5000):
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
                    for _ in range(3000):
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

        return reindex(col)

    def simulated_annealing(col, conf, m, deadline, rng, iterations=200000):
        col = list(col)
        best_col = col[:]
        best_k = k_of(col)
        T = 1.0
        cool = 0.99995
        n_accept = 0
        n_total = 0
        # Track color counts for O(1) k_of
        color_counts = [0] * (best_k + 1)
        for c in col:
            if c < len(color_counts):
                color_counts[c] += 1
            else:
                color_counts.extend([0] * (c - len(color_counts) + 1))
                color_counts[c] += 1
        
        for it in range(iterations):
            if time.time() > deadline:
                break
            n_total += 1
            e = rng.randrange(m)
            c_old = col[e]
            # Pick a random new colour (biased towards lower colours)
            r = rng.random()
            if r < 0.5:
                new_c = rng.randrange(c_old) if c_old > 0 else 0
            else:
                new_c = rng.randrange(best_k + 2)
            
            if new_c == c_old:
                continue
            
            # Check validity
            valid = True
            for j in conf[e]:
                if col[j] == new_c:
                    valid = False
                    break
            
            if valid:
                col[e] = new_c
                # Update color counts
                color_counts[c_old] -= 1
                if new_c >= len(color_counts):
                    color_counts.extend([0] * (new_c - len(color_counts) + 1))
                color_counts[new_c] += 1
                
                # Compute new_k in O(1)
                new_k = best_k
                while new_k > 0 and color_counts[new_k] == 0:
                    new_k -= 1
                
                if new_k < best_k:
                    best_col = col[:]
                    best_k = new_k
                    n_accept += 1
                    T = max(0.01, T * 0.99)
                else:
                    delta = new_k - best_k
                    if delta == 0:
                        if rng.random() < 0.3:
                            n_accept += 1
                        else:
                            col[e] = c_old
                            color_counts[c_old] -= 1
                            color_counts[new_c] += 1
                    else:
                        if rng.random() < (2.718281828 ** (-delta / T)):
                            n_accept += 1
                        else:
                            col[e] = c_old
                            color_counts[c_old] -= 1
                            color_counts[new_c] += 1
                T = max(0.01, T * cool)
        
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
        for trial in range(5):
            if time.time() > t0 + time_budget:
                break
            rng = Random(seed + 1000 * trial)
            col = dsatur_coloring(conf, m, t0 + time_budget)
            col = local_improve(col, conf, m, t0 + time_budget, rng)
            k = k_of(col)
            if k < best_num_colors:
                best_num_colors = k
                best_colors = col[:]
        
        # Phase 2: Simulated annealing from best solution
        if best_colors is not None and time.time() < t0 + time_budget:
            rng = Random(seed + 99999)
            sa_col = simulated_annealing(best_colors, conf, m, t0 + time_budget, rng, iterations=50000)
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

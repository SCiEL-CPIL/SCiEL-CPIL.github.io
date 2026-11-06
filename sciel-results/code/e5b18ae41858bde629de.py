import time
from random import Random
from collections import defaultdict


def construct():
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

    def greedy(col, order, conf):
        m = len(order)
        col = [-1] * m
        for i in order:
            taken = set()
            for j in conf[i]:
                c = col[j]
                if c >= 0:
                    taken.add(c)
            c = 0
            while c in taken:
                c += 1
            col[i] = c
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
        # Maintain color classes incrementally
        cls = build_classes(col)
        color_count = len(cls)
        
        while improved and time.time() < deadline and guard < 500:
            improved = False
            guard += 1
            
            # --- merge pass: fold one whole class into another when compatible ---
            while time.time() < deadline and color_count >= 2:
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
                            color_count -= 1
                            merged = True
                            improved = True
                            break
                    if merged:
                        break
                if not merged:
                    break

            # --- single-edge moves: try to move an edge to any valid lower colour ---
            if not improved and time.time() < deadline and color_count >= 2:
                moved = False
                for _ in range(5000):
                    e = rng.randrange(m)
                    c = col[e]
                    for new_c in range(c):
                        if is_valid_move(col, e, new_c, conf):
                            col[e] = new_c
                            cls[c].discard(e)
                            if not cls[c]:
                                del cls[c]
                                color_count -= 1
                            cls[new_c].add(e)
                            moved = True
                            break
                    if moved:
                        break
                if moved:
                    improved = True

            # --- 2-swap moves ---
            if not improved and time.time() < deadline and color_count >= 2:
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
        best = list(col)
        best_k = k_of(col)
        current = list(col)
        current_k = k_of(current)
        no_improve = 0
        cls = build_classes(current)
        # Track which color is the max to compute k in O(1)
        max_color = best_k - 1
        
        for it in range(iterations):
            if time.time() > deadline:
                break
            if no_improve > 20000:
                break
            
            cols = sorted(cls.keys())
            if len(cols) < 2:
                break
            
            e = rng.randrange(m)
            c = current[e]
            
            # Try to move to a different valid color, biased toward lower
            if rng.random() < 0.75:
                # Prefer lower colors
                lower = [c2 for c2 in cols if c2 < c]
                if lower:
                    new_c = rng.choice(lower)
                elif cols:
                    others = [c2 for c2 in cols if c2 != c]
                    if others:
                        new_c = rng.choice(others)
                    else:
                        continue
                else:
                    continue
            else:
                others = [c2 for c2 in cols if c2 != c]
                if not others:
                    continue
                new_c = rng.choice(others)
            
            if is_valid_move(current, e, new_c, conf):
                cls[c].discard(e)
                old_size = len(cls[c])
                if not cls[c]:
                    del cls[c]
                cls[new_c].add(e)
                current[e] = new_c
                
                # Update k: if we freed the max color, recompute
                if c == max_color and old_size == 1:
                    # Max color freed, find new max
                    new_max = max(cls.keys()) if cls else 0
                    max_color = new_max
                    current_k = max_color + 1
                elif new_c > max_color:
                    max_color = new_c
                    current_k = max_color + 1
                
                if current_k < best_k:
                    best_k = current_k
                    best = list(current)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        
        return reindex(best)

    results = []
    for idx, (n, d, seed) in enumerate(((40, 4, 101), (60, 5, 102), (80, 6, 103))):
        edges = gen_instance(n, d, seed)
        m = len(edges)
        if m == 0:
            results.append([])
            continue
        conf = build_conflict(edges)
        cdeg = [len(c) for c in conf]
        order_base = list(range(m))
        deg_order = sorted(order_base, key=lambda i: -cdeg[i])

        best = None
        best_k = m + 1

        def consider(col):
            nonlocal best, best_k
            k = k_of(col)
            if k < best_k:
                best_k = k
                best = list(col)

        # DSATUR initial coloring
        ds_deadline = time.time() + 5.0
        consider(dsatur_coloring(conf, m, ds_deadline))
        
        # a few deterministic seeds
        consider(greedy([None] * m, order_base, conf))
        consider(greedy([None] * m, deg_order, conf))
        consider(greedy([None] * m, sorted(order_base, key=lambda i: cdeg[i]), conf))

        rng = Random(999 + seed)
        remaining = 3 - idx
        # Allocate more time to later (larger) instances
        restart_deadline = DEADLINE - 15.0 * remaining

        # Randomized restarts with greedy (fewer, since DSATUR is strong)
        max_restarts = 50
        restart_count = 0
        while time.time() < restart_deadline and restart_count < max_restarts:
            order = list(order_base)
            rng.shuffle(order)
            consider(greedy([None] * m, order, conf))
            restart_count += 1

        # Local compression on the best
        ls_deadline = min(restart_deadline + 20.0, DEADLINE - 15.0 * (remaining - 1))
        if best is not None:
            best = local_improve(best, conf, m, ls_deadline, rng)
            consider(best)
            if time.time() < ls_deadline:
                best = local_improve(best, conf, m, ls_deadline, rng)
                consider(best)

        # Simulated annealing for further improvement
        sa_deadline = DEADLINE - 10.0 * (remaining - 1)
        if best is not None and time.time() < sa_deadline:
            sa_result = simulated_annealing(best, conf, m, sa_deadline, rng, iterations=300000)
            consider(sa_result)
            if time.time() < sa_deadline:
                sa_result2 = simulated_annealing(best, conf, m, sa_deadline, rng, iterations=300000)
                consider(sa_result2)
            # Final local polish
            if best is not None and time.time() < sa_deadline:
                final_deadline = DEADLINE - 5.0 * (remaining - 1)
                best = local_improve(best, conf, m, final_deadline, rng)
                consider(best)

        if best is None:
            best = greedy([None] * m, order_base, conf)
        results.append(best)
    return results

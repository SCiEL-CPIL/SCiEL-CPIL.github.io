import time
from random import Random


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

    def build(edges):
        m = len(edges)
        if m == 0:
            return 0, [], [], []
        n = max(max(u, v) for u, v in edges) + 1
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        inc = [set() for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            inc[u].add(i)
            inc[v].add(i)
        # conflict[e] = edges f != e with an endpoint in adj[u] | adj[v]  (e=(u,v))
        conflict = [set() for _ in range(m)]
        for i, (u, v) in enumerate(edges):
            forb = adj[u] | adj[v]
            s = set()
            for w in forb:
                s |= inc[w]
            s.discard(i)
            conflict[i] = s
        return n, adj, inc, conflict

    def greedy(order, conflict, m):
        col = [-1] * m
        for i in order:
            used = set()
            for j in conflict[i]:
                cj = col[j]
                if cj >= 0:
                    used.add(cj)
            c = 0
            while c in used:
                c += 1
            col[i] = c
        return col

    def num_colours(col):
        return len(set(col))

    def is_valid(col, conflict):
        classes = {}
        for i, c in enumerate(col):
            classes.setdefault(c, set()).add(i)
        for es in classes.values():
            for i in es:
                for j in es:
                    if i != j and j in conflict[i]:
                        return False
        return True

    def shadows_of(col, conflict):
        classes = {}
        for i, c in enumerate(col):
            classes.setdefault(c, set()).add(i)
        shadows = {}
        for c, es in classes.items():
            s = set()
            for e in es:
                s |= conflict[e]
            shadows[c] = s
        return classes, shadows

    def local_search(col, conflict, m, deadline):
        col = col[:]
        best = col[:]
        best_k = num_colours(col)
        no_progress = 0
        max_no_progress = 100

        for _ in range(8000):
            if time.time() > deadline:
                break

            classes, shadows = shadows_of(col, conflict)
            ckeys = sorted(classes.keys())

            # Try to merge two colour classes
            merged = False
            for a in ckeys:
                if a not in classes:
                    continue
                for b in ckeys:
                    if a >= b or b not in classes:
                        continue
                    # Check if class b can be merged into class a
                    # No edge in class b can conflict with any edge in class a
                    if shadows[a] & classes[b]:
                        continue
                    if shadows[b] & classes[a]:
                        continue
                    for e in classes[b]:
                        col[e] = a
                    del classes[b]
                    merged = True
                    break
                if merged:
                    break

            if merged:
                no_progress = 0
                k = num_colours(col)
                if k < best_k:
                    best_k = k
                    best = col[:]
                continue

            # Try single-edge moves
            moved = False
            classes, shadows = shadows_of(col, conflict)
            ckeys = sorted(classes.keys())
            for a in ckeys:
                if a not in classes:
                    continue
                for e in list(classes[a]):
                    for d in ckeys:
                        if d == a or d not in classes:
                            continue
                        # Check if edge e can move to class d
                        if e in shadows[d]:
                            continue
                        col[e] = d
                        moved = True
                        break
                    if moved:
                        break
                if moved:
                    break

            if moved:
                no_progress = 0
                k = num_colours(col)
                if k < best_k:
                    best_k = k
                    best = col[:]
                continue

            # Try 2-swaps: exchange edges between two colour classes
            swapped = False
            classes, shadows = shadows_of(col, conflict)
            ckeys = sorted(classes.keys())
            for a in ckeys:
                if a not in classes:
                    continue
                for b in ckeys:
                    if a >= b or b not in classes:
                        continue
                    for e1 in list(classes[a]):
                        for e2 in list(classes[b]):
                            # Check if swapping e1<->e2 is valid
                            ok = True
                            for x in classes[b]:
                                if x == e2:
                                    continue
                                if x in conflict[e1]:
                                    ok = False
                                    break
                            if ok:
                                for x in classes[a]:
                                    if x == e1:
                                        continue
                                    if x in conflict[e2]:
                                        ok = False
                                        break
                            if ok:
                                col[e1] = b
                                col[e2] = a
                                swapped = True
                                break
                        if swapped:
                            break
                    if swapped:
                        break

            if swapped:
                no_progress = 0
                k = num_colours(col)
                if k < best_k:
                    best_k = k
                    best = col[:]
                continue

            # No move found
            no_progress += 1
            if no_progress >= max_no_progress:
                break

        if is_valid(best, conflict):
            return best
        return col

    def solve_instance(edges, deadline):
        m = len(edges)
        n, adj, inc, conflict = build(edges)
        rng = Random(777 + m)
        cd = [len(conflict[i]) for i in range(m)]
        base_orders = [
            list(range(m)),
            sorted(range(m), key=lambda i: -cd[i]),
            sorted(range(m), key=lambda i: cd[i]),
        ]
        cand = []
        seen = set()

        def add_cand(col):
            sig = tuple(col)
            if sig in seen:
                return
            seen.add(sig)
            cand.append((num_colours(col), col[:]))

        for o in base_orders:
            add_cand(greedy(o, conflict, m))

        # Random restarts with greedy
        t_re = time.time() + 1.5
        cnt = 0
        while time.time() < t_re and cnt < 300:
            o = list(range(m))
            rng.shuffle(o)
            add_cand(greedy(o, conflict, m))
            cnt += 1

        cand.sort(key=lambda x: x[0])
        top = []
        tsig = set()
        for k, col in cand:
            s = tuple(col)
            if s in tsig:
                continue
            tsig.add(s)
            top.append((k, col))
            if len(top) >= 8:
                break

        best = None
        best_k = 10 ** 9
        n_runs = len(top)
        for k, col in top:
            if time.time() > deadline - 0.5:
                break
            slice_dl = min(deadline - 0.3, time.time() + max(0.5, (deadline - time.time()) / max(1, n_runs)))
            res = local_search(col, conflict, m, slice_dl)
            rk = num_colours(res)
            if rk < best_k:
                best_k = rk
                best = res
            n_runs -= 1

        if best is None:
            best = top[0][1]
        if not is_valid(best, conflict):
            best = greedy(list(range(m)), conflict, m)
        present = sorted(set(best))
        remap = {old: new for new, old in enumerate(present)}
        return [remap[c] for c in best]

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    out = []
    for idx, spec in enumerate(specs):
        edges = gen_instance(*spec)
        m = len(edges)
        left_after = len(specs) - 1 - idx
        inst_dl = min(DEADLINE - 3.0 - 2.0 * left_after, time.time() + 40.0)
        if inst_dl < time.time() + 4.0:
            inst_dl = min(DEADLINE - 1.0, time.time() + 4.0)
        try:
            out.append(solve_instance(edges, inst_dl))
        except Exception:
            # fast, always-valid fallback
            n, adj, inc, conflict = build(edges)
            col = greedy(list(range(m)), conflict, m)
            present = sorted(set(col))
            remap = {old: new for new, old in enumerate(present)}
            out.append([remap[c] for c in col])
    return out

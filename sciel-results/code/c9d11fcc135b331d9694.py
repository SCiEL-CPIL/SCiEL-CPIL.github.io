import time, random, math

def construct():
    start = time.time()
    DEADLINE = start + 78.0
    N = 180

    # Build all triples (a<b<c) with 1/a = 1/b + 1/c  <=>  (b-a)(c-a)=a^2
    triples = []
    seen = set()
    for a in range(1, N + 1):
        a2 = a * a
        d = 1
        while d * d <= a2:
            if a2 % d == 0:
                e = a2 // d
                if d != e:
                    b = a + d
                    c = a + e
                    if b <= N and c <= N:
                        t = (a, b, c) if b < c else (a, c, b)
                        if t not in seen:
                            seen.add(t)
                            triples.append(t)
            d += 1

    adj = [[] for _ in range(N + 1)]
    for ti, (a, b, c) in enumerate(triples):
        adj[a].append((b, c, ti))
        adj[b].append((a, c, ti))
        adj[c].append((a, b, ti))

    # Precompute for each vertex v: list of (u, w, tri_idx) so we can track conflict counts
    # conflict_count[tri_idx] = number of vertices in triple that are present (0..3)
    # A triple is "active" when conflict_count == 3

    def make_present(lo, hi):
        p = [False] * (N + 1)
        for v in range(lo, hi + 1):
            p[v] = True
        return p

    def build_conflict_counts(p):
        cc = [0] * len(triples)
        for i, (a, b, c) in enumerate(triples):
            cnt = 0
            if p[a]: cnt += 1
            if p[b]: cnt += 1
            if p[c]: cnt += 1
            cc[i] = cnt
        return cc

    def count_active_from_cc(cc):
        return sum(1 for x in cc if x == 3)

    def greedy_grow(p):
        p = p[:]
        cc = build_conflict_counts(p)
        improved = True
        while improved:
            improved = False
            for v in range(1, N + 1):
                if not p[v]:
                    # Check if adding v would create any active triple
                    safe = True
                    for (u, w, ti) in adj[v]:
                        if p[u] and p[w]:
                            safe = False
                            break
                    if safe:
                        p[v] = True
                        for (u, w, ti) in adj[v]:
                            cc[ti] += 1
                        improved = True
        return p

    def local_search_remove_then_add(p):
        """Remove elements to eliminate conflicts, then greedily add back."""
        p = p[:]
        cc = build_conflict_counts(p)
        # Phase 1: remove elements that are in active triples, preferring those in most active triples
        changed = True
        while changed:
            changed = False
            # Count active triples per vertex
            act_count = [0] * (N + 1)
            for i, (a, b, c) in enumerate(triples):
                if cc[i] == 3:
                    act_count[a] += 1
                    act_count[b] += 1
                    act_count[c] += 1
            # Find vertex with most active triples
            max_cnt = 0
            max_v = -1
            for v in range(1, N + 1):
                if p[v] and act_count[v] > max_cnt:
                    max_cnt = act_count[v]
                    max_v = v
            if max_v > 0:
                p[max_v] = False
                for (u, w, ti) in adj[max_v]:
                    cc[ti] -= 1
                changed = True
        # Phase 2: greedily add back
        improved = True
        while improved:
            improved = False
            for v in range(1, N + 1):
                if not p[v]:
                    safe = True
                    for (u, w, ti) in adj[v]:
                        if p[u] and p[w]:
                            safe = False
                            break
                    if safe:
                        p[v] = True
                        for (u, w, ti) in adj[v]:
                            cc[ti] += 1
                        improved = True
        return p

    # Try multiple starting points
    starts = []
    for lo in [70, 80, 90, 91, 100, 110]:
        p = make_present(lo, 180)
        starts.append(p)
    # Also try a greedy construction from scratch
    p_greedy = [False] * (N + 1)
    for v in range(1, N + 1):
        safe = True
        for (u, w, ti) in adj[v]:
            if p_greedy[u] and p_greedy[w]:
                safe = False
                break
        if safe:
            p_greedy[v] = True
    starts.append(p_greedy)

    # Evaluate all starts and their grown versions
    best_size = 0
    best_set = None
    for p in starts:
        g = greedy_grow(p)
        cc = build_conflict_counts(g)
        act = count_active_from_cc(cc)
        sz = sum(g[1:])
        if act == 0 and sz > best_size:
            best_size = sz
            best_set = g[:]
        # Also try local search
        ls = local_search_remove_then_add(p)
        cc2 = build_conflict_counts(ls)
        act2 = count_active_from_cc(cc2)
        sz2 = sum(ls[1:])
        if act2 == 0 and sz2 > best_size:
            best_size = sz2
            best_set = ls[:]

    if best_set is None:
        best_set = make_present(91, 180)
        best_size = sum(best_set[1:])

    # Simulated annealing with conflict-count tracking
    random.seed(12345)
    present = best_set[:]
    cc = build_conflict_counts(present)
    cur_active = count_active_from_cc(cc)
    cur_size = sum(present[1:])

    T = 5.0
    TMIN = 0.01
    while time.time() < DEADLINE and T > TMIN:
        v = random.randint(1, N)
        old = present[v]
        dact = 0
        for (u, w, ti) in adj[v]:
            if present[u] and present[w]:
                if not old:
                    dact += 1
                else:
                    dact -= 1
        dsize = 1 if not old else -1
        ds = dsize - 100000 * dact
        if ds >= 0 or random.random() < math.exp(ds / T):
            present[v] = not old
            for (u, w, ti) in adj[v]:
                cc[ti] += 1 if not old else -1
            cur_active += dact
            cur_size += dsize
            if cur_active == 0 and cur_size > best_size:
                best_size = cur_size
                best_set = present[:]
        T *= 0.9998

    # Final polish: greedy grow
    if time.time() < DEADLINE:
        g = greedy_grow(best_set)
        cc_final = build_conflict_counts(g)
        if count_active_from_cc(cc_final) == 0 and sum(g[1:]) > best_size:
            best_size = sum(g[1:])
            best_set = g

    # Another local search pass
    if time.time() < DEADLINE:
        ls = local_search_remove_then_add(best_set)
        cc_ls = build_conflict_counts(ls)
        if count_active_from_cc(cc_ls) == 0 and sum(ls[1:]) > best_size:
            best_size = sum(ls[1:])
            best_set = ls

    res = [v for v in range(1, N + 1) if best_set[v]]
    if not res:
        res = list(range(91, 181))
    return res

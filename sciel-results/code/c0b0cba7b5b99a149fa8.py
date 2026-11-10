def construct():
    import time
    from random import Random
    import numpy as np

    t0 = time.time()
    DEADLINE = 135.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return cross_edges

    def build_adj(n, k, cross_edges):
        adj = np.zeros((n, n), dtype=bool)
        for u in range(k):
            for w in range(u + 1, k):
                adj[u, w] = adj[w, u] = True
        for u, v in cross_edges:
            adj[u, v] = adj[v, u] = True
        return adj

    def is_clique(S, adj):
        S = list(S)
        for i in range(len(S)):
            a = S[i]
            row = adj[a]
            for j in range(i + 1, len(S)):
                if not row[S[j]]:
                    return False
        return True

    def maximal_clique_greedy(cand, adj):
        cand = list(cand)
        if not cand:
            return []
        order = sorted(cand, key=lambda v: -int(adj[v].sum()))
        S = []
        for v in order:
            if all(adj[v, s] for s in S):
                S.append(v)
        return S

    def bron_kerbosch_max(cand, adj):
        # Return the largest clique found via Bron-Kerbosch (first maximal that is largest)
        best = []
        def bk(R, P):
            nonlocal best
            if len(R) <= len(best):
                return
            if not P:
                if len(R) > len(best):
                    best = list(R)
                return
            # pivot
            pivot = max(P, key=lambda v: int(adj[v].sum()))
            Q = [v for v in P if not adj[v, pivot]]
            for v in Q:
                R.append(v)
                newP = [w for w in P if adj[v, w]]
                bk(R, newP)
                R.pop()
                P = [w for w in P if w != v]
                if len(R) <= len(best):
                    break
        bk([], list(cand))
        return best

    def maximal_clique(cand, adj, use_bk):
        g = maximal_clique_greedy(cand, adj)
        if use_bk and len(cand) <= 400:
            b = bron_kerbosch_max(cand, adj)
            if len(b) > len(g):
                return b
        return g

    def initial_partition(n, adj, rng, use_bk):
        remaining = set(range(n))
        cliques = []
        while remaining:
            if time.time() > DEADLINE:
                break
            # candidate: a random seed to diversify
            seed_v = rng.choice(sorted(remaining))
            # grow a maximal clique containing seed_v
            cand = [seed_v]
            for w in sorted(remaining, key=lambda x: -int(adj[seed_v, x])):
                if w != seed_v and adj[seed_v, w]:
                    cand.append(w)
            # refine to maximal clique
            S = [seed_v]
            order = [w for w in cand if w != seed_v]
            order.sort(key=lambda v: -int(adj[v].sum()))
            for v in order:
                if all(adj[v, s] for s in S):
                    S.append(v)
            if use_bk and len(cand) <= 400:
                b = bron_kerbosch_max(cand, adj)
                if len(b) > len(S):
                    S = b
            if not S:
                # should not happen
                S = [seed_v]
            cliques.append(S)
            for s in S:
                remaining.discard(s)
        return cliques

    def edge_count(clique, adj):
        m = len(clique)
        return m * (m - 1) // 2

    def total_edges(n, adj):
        return int(adj.sum()) // 2

    def build_coverage(cliques, adj, n):
        # coverage[e] = number of cliques containing both endpoints
        cov = np.zeros((n, n), dtype=np.int32)
        for S in cliques:
            for i in range(len(S)):
                a = S[i]
                for j in range(i + 1, len(S)):
                    b = S[j]
                    cov[a, b] += 1
                    cov[b, a] += 1
        return cov

    def local_search(cliques, adj, n, rng, time_budget):
        # Try to reduce number of cliques by moving vertices and deleting redundant cliques.
        # A clique is redundant if every edge it contains is covered by at least one other clique.
        # We represent cliques as lists; maintain cov.
        cov = build_coverage(cliques, adj, n)
        E = total_edges(n, adj)
        # Precompute for each edge the set of cliques that contain it? Instead, on deletion we recompute cov for that clique's edges.

        def remove_clique(idx):
            S = cliques[idx]
            for i in range(len(S)):
                a = S[i]
                for j in range(i + 1, len(S)):
                    b = S[j]
                    cov[a, b] -= 1
                    cov[b, a] -= 1
            cliques.pop(idx)

        def add_clique(S):
            cliques.append(S)
            for i in range(len(S)):
                a = S[i]
                for j in range(i + 1, len(S)):
                    b = S[j]
                    cov[a, b] += 1
                    cov[b, a] += 1

        def is_redundant(idx):
            S = cliques[idx]
            for i in range(len(S)):
                a = S[i]
                for j in range(i + 1, len(S)):
                    b = S[j]
                    if cov[a, b] <= 1:
                        return False
            return True

        # First pass: delete any redundant cliques
        changed = True
        while changed:
            changed = False
            for idx in range(len(cliques) - 1, -1, -1):
                if len(cliques[idx]) < 2:
                    remove_clique(idx)
                    changed = True
                elif is_redundant(idx):
                    remove_clique(idx)
                    changed = True

        # Move-based local search: pick a vertex from a clique, try to move it to another clique
        # where it is adjacent to all, and after moving the source clique is still valid (clique) and
        # ideally becomes redundant (then delete). We do random restarts within time budget.
        best_count = len(cliques)
        best_cliques = [list(c) for c in cliques]

        t_start = time.time()
        while time.time() - t_start < time_budget:
            if time.time() > DEADLINE:
                break
            # random restart: shuffle vertex order inside cliques and try moves
            for _ in range(200):
                if time.time() - t_start > time_budget or time.time() > DEADLINE:
                    break
                # pick a random clique with >=3 vertices (moving a vertex out)
                if len(cliques) < 2:
                    break
                cand_sources = [i for i, S in enumerate(cliques) if len(S) >= 3]
                if not cand_sources:
                    break
                idx = rng.choice(cand_sources)
                S = cliques[idx]
                v = rng.choice(S)
                # find target clique T (!= idx) where v is adjacent to all of T
                # and T is a clique (it is), and adding v keeps clique
                best_target = None
                best_gain = 0
                for j, T in enumerate(cliques):
                    if j == idx or len(T) < 2:
                        continue
                    if all(adj[v, t] for t in T):
                        # moving v to T: source S' = S \ {v}; if S' has <2, it's deleted
                        # gain: if S' becomes redundant (or empty) we delete one clique
                        # Heuristic: prefer T with larger size
                        gain = len(T)
                        if len(S) - 1 < 2:
                            gain += 1000
                        if gain > best_gain:
                            best_gain = gain
                            best_target = j
                if best_target is None:
                    continue
                # Perform move
                T = cliques[best_target]
                # remove v from S
                S.remove(v)
                T.append(v)
                # Update cov: v's edges with S\{v} removed from S coverage, added to T coverage
                # Easier: recompute cov for S and T
                # Remove old contributions
                old_S = S + [v]
                for i in range(len(old_S)):
                    a = old_S[i]
                    for j in range(i + 1, len(old_S)):
                        b = old_S[j]
                        cov[a, b] -= 1
                        cov[b, a] -= 1
                # Add new S
                for i in range(len(S)):
                    a = S[i]
                    for j in range(i + 1, len(S)):
                        b = S[j]
                        cov[a, b] += 1
                        cov[b, a] += 1
                # Remove old T
                old_T = [t for t in T if t != v]
                for i in range(len(old_T)):
                    a = old_T[i]
                    for j in range(i + 1, len(old_T)):
                        b = old_T[j]
                        cov[a, b] -= 1
                        cov[b, a] -= 1
                # Add new T
                for i in range(len(T)):
                    a = T[i]
                    for j in range(i + 1, len(T)):
                        b = T[j]
                        cov[a, b] += 1
                        cov[b, a] += 1
                # Now delete redundant cliques
                for k in range(len(cliques) - 1, -1, -1):
                    if len(cliques[k]) < 2:
                        remove_clique(k)
                    elif is_redundant(k):
                        remove_clique(k)
                if len(cliques) < best_count:
                    best_count = len(cliques)
                    best_cliques = [list(c) for c in cliques]

        # Restore best
        cliques[:] = [list(c) for c in best_cliques]
        return cliques

    def solve_instance(n, k, p, seed):
        cross_edges = instance(n, k, p, seed)
        adj = build_adj(n, k, cross_edges)
        rng = Random(seed * 7919 + 13)

        # Initial partitions with a few random seeds
        best = None
        best_count = 10**9
        for attempt in range(6):
            if time.time() > DEADLINE:
                break
            use_bk = (attempt >= 2 and n <= 90)
            cliques = initial_partition(n, adj, rng, use_bk)
            if len(cliques) < best_count:
                best_count = len(cliques)
                best = [list(c) for c in cliques]
        if best is None:
            best = initial_partition(n, adj, rng, False)
            best_count = len(best)

        # Local search improvement
        remaining = DEADLINE - time.time()
        if remaining > 5:
            best = local_search(best, adj, n, rng, max(1.0, remaining - 5))

        # ILP polish if PuLP available and small enough
        if time.time() < DEADLINE - 10:
            try:
                import pulp
                cliques = best
                m = len(cliques)
                E = total_edges(n, adj)
                # Build edge list
                edge_list = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u, v]:
                            edge_list.append((u, v))
                if m <= 60 and len(edge_list) <= 8000:
                    prob = pulp.LpProblem("clique_cover", pulp.LpMinimize)
                    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(m)]
                    prob += pulp.lpSum(x)
                    for (u, v) in edge_list:
                        # find cliques containing both u and v
                        cover = []
                        for i, S in enumerate(cliques):
                            if u in S and v in S:
                                cover.append(i)
                        if cover:
                            prob += pulp.lpSum([x[i] for i in cover]) >= 1
                    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=int(max(1, DEADLINE - time.time() - 5))))
                    if pulp.LpStatus[prob.status] == "Optimal":
                        new_cliques = [cliques[i] for i in range(m) if x[i].varValue and x[i].varValue > 0.5]
                        if new_cliques and len(new_cliques) < len(best):
                            best = new_cliques
            except Exception:
                pass

        # Final validation & repair: ensure every edge is covered; if not, add missing as 2-cliques
        cov = build_coverage(best, adj, n)
        missing = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v] and cov[u, v] < 1:
                    missing.append((u, v))
        for (u, v) in missing:
            best.append([u, v])

        # Ensure each clique is a valid clique (it should be) and has >=2 vertices
        cleaned = []
        for S in best:
            S = list(S)
            if len(S) < 2:
                continue
            # verify
            if is_clique(S, adj):
                cleaned.append(S)
            else:
                # split into edges as fallback
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        if adj[S[i], S[j]]:
                            cleaned.append([S[i], S[j]])
        # Re-check coverage after cleaning; add any missing
        cov = build_coverage(cleaned, adj, n)
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v] and cov[u, v] < 1:
                    cleaned.append([u, v])
        return cleaned

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    for (n, k, p, seed) in specs:
        if time.time() > DEADLINE:
            # Fallback: trivial valid partition
            ce = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in ce:
                cliques.append([int(u), int(v)])
            result.append(cliques)
        else:
            result.append(solve_instance(n, k, p, seed))

    # Ensure types are Python int and lists
    final = []
    for partition in result:
        clean = []
        for clique in partition:
            clean.append([int(x) for x in clique])
        final.append(clean)
    return final

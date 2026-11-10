def construct():
    import time
    from random import Random

    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Build cross edges
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return cross_edges

    def solve_instance(n, k, p, seed, variant_seed=None):
        cross_edges = instance(n, k, p, seed)

        # N[v] is list of neighbors in C for v in I
        N = {v: [] for v in range(k, n)}
        for u, v in cross_edges:
            N[v].append(u)

        # Available C-edges as frozensets (C is a clique, so every pair is an edge)
        all_c_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_c_edges.add(frozenset((u, w)))
        used = set()

        # Fast O(1) degree counter for remaining C-edges
        deg = [0] * k
        for u in range(k):
            for w in range(u + 1, k):
                deg[u] += 1
                deg[w] += 1

        def mark_used(a, b):
            used.add(frozenset((a, b)))
            deg[a] -= 1
            deg[b] -= 1

        mixed_cliques = []

        # Determine order of I vertices
        if variant_seed is None:
            I_vertices = sorted(range(k, n), key=lambda x: len(N[x]))
        else:
            rng_var = Random(variant_seed)
            if variant_seed % 3 == 0:
                I_vertices = sorted(range(k, n), key=lambda x: len(N[x]))
            elif variant_seed % 3 == 1:
                I_vertices = sorted(range(k, n), key=lambda x: len(N[x]) + rng_var.random() * 0.5)
            else:
                I_vertices = list(range(k, n))
                rng_var.shuffle(I_vertices)

        for v in I_vertices:
            neighbors = N[v]
            if not neighbors:
                continue

            rem_set = set(neighbors)

            # Sort neighbors by current remaining degree (ascending) to consume
            # low-degree (cheap) C-vertices first.
            sorted_rem = sorted(rem_set, key=lambda c: deg[c])
            matched = set()
            pairings = []

            for idx_a in range(len(sorted_rem)):
                c_a = sorted_rem[idx_a]
                if c_a in matched:
                    continue
                best_partner = None
                best_cost = None
                for idx_b in range(idx_a + 1, len(sorted_rem)):
                    c_b = sorted_rem[idx_b]
                    if c_b in matched:
                        continue
                    e = frozenset((c_a, c_b))
                    if e in used:
                        continue
                    cost = deg[c_a] + deg[c_b]
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_partner = c_b
                if best_partner is not None:
                    matched.add(c_a)
                    matched.add(best_partner)
                    pairings.append((c_a, best_partner))
                    mark_used(c_a, best_partner)

            for c_a, c_b in pairings:
                mixed_cliques.append([int(v), int(c_a), int(c_b)])

            for c in sorted_rem:
                if c not in matched:
                    mixed_cliques.append([int(v), int(c)])

        # Cover remaining C-edges with pure-C cliques (greedy maximal cliques)
        pure_cliques = []
        rem_edges = set()
        for e in all_c_edges:
            if e not in used:
                rem_edges.add(e)

        while rem_edges:
            if time.time() > deadline - 1.0:
                for e in rem_edges:
                    a, b = tuple(e)
                    pure_cliques.append([int(a), int(b)])
                break

            # Pick the edge with max combined remaining degree as a seed
            best_edge = None
            best_score = -1
            for e in rem_edges:
                a, b = tuple(e)
                score = deg[a] + deg[b]
                if score > best_score:
                    best_score = score
                    best_edge = e

            if best_edge is None:
                break

            a0, b0 = tuple(best_edge)

            # Build a greedy maximal clique from the seed edge.
            S = [a0, b0]
            S_set = {a0, b0}

            cand = set()
            for e in rem_edges:
                for x in e:
                    if x not in S_set:
                        cand.add(x)

            while cand:
                best_v = None
                best_deg = -1
                for w in cand:
                    ok = True
                    d = 0
                    for s in S:
                        if frozenset((w, s)) not in rem_edges:
                            ok = False
                            break
                        d += 1
                    if ok and d > best_deg:
                        best_deg = d
                        best_v = w
                if best_v is None:
                    break
                S.append(best_v)
                S_set.add(best_v)
                new_cand = set()
                for w in cand:
                    if w == best_v:
                        continue
                    ok = True
                    for s in S:
                        if frozenset((w, s)) not in rem_edges:
                            ok = False
                            break
                    if ok:
                        new_cand.add(w)
                cand = new_cand

            if len(S) >= 2:
                pure_cliques.append([int(x) for x in S])
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        e = frozenset((S[i], S[j]))
                        if e in rem_edges:
                            rem_edges.discard(e)
                            deg[S[i]] -= 1
                            deg[S[j]] -= 1
            else:
                pure_cliques.append([int(a0), int(b0)])
                if best_edge in rem_edges:
                    rem_edges.discard(best_edge)
                    deg[a0] -= 1
                    deg[b0] -= 1

        return mixed_cliques + pure_cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []

    for (n, k, p, seed) in specs:
        best_cliques = None
        best_count = float('inf')

        if time.time() > deadline - 5.0:
            ce = instance(n, k, p, seed)
            cliques = [[int(u) for u in range(k)]]
            for (u, v) in ce:
                cliques.append([int(u), int(v)])
            result.append(cliques)
            continue

        # Try base first
        cliques = solve_instance(n, k, p, seed, None)
        count = len(cliques)
        if count < best_count:
            best_count = count
            best_cliques = cliques

        # Try variants
        for i in range(40):
            if time.time() > deadline - 15.0:
                break
            v_seed = seed + 1000 + i
            cliques = solve_instance(n, k, p, seed, v_seed)
            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = cliques

        result.append(best_cliques)

    # Ensure types
    final = []
    for partition in result:
        clean = []
        for clique in partition:
            clean.append([int(x) for x in clique])
        final.append(clean)
    return final

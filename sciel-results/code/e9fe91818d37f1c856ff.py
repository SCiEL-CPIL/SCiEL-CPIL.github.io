import time
import random

def construct():
    start_time = time.time()
    TOTAL_BUDGET = 140  # safety margin from 150s wall-clock limit

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        adj_mask = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        return adj_mask

    def solve_instance(n, p, seed, time_budget):
        adj_mask_orig = get_graph(n, p, seed)

        if not any(adj_mask_orig):
            return []

        def iter_bits(x):
            while x:
                lsb = x & -x
                yield lsb.bit_length() - 1
                x ^= lsb

        def edge_common(adj, n):
            """List (common, u, v) for every remaining edge."""
            edges = []
            for u in range(n):
                m = adj[u]
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if v > u:
                        edges.append(((adj[u] & adj[v]).bit_count(), u, v))
                    m ^= lsb
            return edges

        def pick_edge(edges, rng, mode):
            """Select a seed edge.
            mode 0: min common (deterministic / soft-min)
            mode 1: max common (dense region first)
            mode 2: uniform random
            """
            if not edges:
                return -1, -1
            if mode == 2:
                c, u, v = edges[rng.randint(0, len(edges) - 1)]
                return u, v
            if mode == 1:
                edges.sort(key=lambda x: (-x[0], x[1], x[2]))
            else:
                edges.sort(key=lambda x: (x[0], x[1], x[2]))
            if rng is None:
                return edges[0][1], edges[0][2]
            # Soft selection over the near-extreme set to add exploration.
            k = min(len(edges), max(8, len(edges) // 4))
            best_key = float('inf')
            best_u = best_v = -1
            for common, u, v in edges[:k]:
                key = common + rng.random()
                if key < best_key:
                    best_key = key
                    best_u, best_v = u, v
            return best_u, best_v

        def expand_clique(seed_u, seed_v, adj, rng=None):
            C = [seed_u, seed_v]
            U = adj[seed_u] & adj[seed_v]
            while U:
                best_u = -1
                best_score = -1
                best_score2 = -1
                candidates = list(iter_bits(U))
                if rng is not None and len(candidates) > 1:
                    rng.shuffle(candidates)
                for u in candidates:
                    score = (adj[u] & U).bit_count()
                    score2 = adj[u].bit_count()
                    if score > best_score or (score == best_score and score2 > best_score2):
                        best_score = score
                        best_score2 = score2
                        best_u = u
                if best_u == -1:
                    break
                C.append(best_u)
                U &= adj[best_u]
            return C

        def remove_clique_edges(C, adj):
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v = C[i], C[j]
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

        def is_clique_union(C1, C2, adj):
            for u in C1:
                for v in C2:
                    if not (adj[u] & (1 << v)):
                        return False
            return True

        def is_clique(C, adj):
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v = C[i], C[j]
                    if not (adj[u] & (1 << v)):
                        return False
            return True

        def try_merge(cliques, adj, n, deadline):
            """Merge cliques whose union is a clique; also try best-vertex reassignment."""
            changed = True
            while changed and (time.time() < deadline):
                changed = False
                # Phase A: union merges
                new_cliques = []
                i = 0
                while i < len(cliques):
                    C_i = cliques[i]
                    merged = False
                    for j in range(i + 1, len(cliques)):
                        C_j = cliques[j]
                        if is_clique_union(C_i, C_j, adj):
                            new_cliques = [C_i + C_j] + [cliques[k] for k in range(i + 1, len(cliques)) if k != j]
                            i = 0
                            changed = True
                            merged = True
                            break
                    if not merged:
                        new_cliques.append(C_i)
                        i += 1
                cliques = new_cliques
                if not changed:
                    break
                # Phase B: best-vertex reassignment (drop a 2-clique by absorbing a vertex)
                if time.time() < deadline:
                    for a in range(len(cliques)):
                        if time.time() >= deadline:
                            break
                        Ca = cliques[a]
                        if len(Ca) < 2:
                            continue
                        for b in range(len(cliques)):
                            if a == b:
                                continue
                            Cb = cliques[b]
                            for w in list(Ca):
                                rest_a = [x for x in Ca if x != w]
                                new_b = Cb + [w]
                                if len(rest_a) >= 2 and is_clique(rest_a, adj) and is_clique(new_b, adj):
                                    if len(Ca) == 2:
                                        cliques = [cliques[k] for k in range(len(cliques)) if k != a]
                                        cliques[b] = new_b
                                        changed = True
                                        break
                            if changed:
                                break
                        if changed:
                            break
            return cliques

        def greedy_partition(adj_mask_orig, n, rng=None, deadline=None, mode=0):
            if deadline is None:
                deadline = time.time() + 10
            adj = list(adj_mask_orig)
            cliques = []
            while True:
                if time.time() > deadline:
                    break
                edges = edge_common(adj, n)
                u, v = pick_edge(edges, rng, mode)
                if u == -1:
                    break
                C = expand_clique(u, v, adj, rng)
                if len(C) < 2:
                    C = [u, v]
                cliques.append(C)
                remove_clique_edges(C, adj)
            cliques = try_merge(cliques, adj, n, deadline)
            return cliques

        def validate(cliques, adj_mask_orig, n):
            for C in cliques:
                if len(C) < 2:
                    return False
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v = C[i], C[j]
                        if not (adj_mask_orig[u] & (1 << v)):
                            return False
            edge_set = set()
            for C in cliques:
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v = C[i], C[j]
                        key = (min(u, v), max(u, v))
                        if key in edge_set:
                            return False
                        edge_set.add(key)
            for u in range(n):
                m = adj_mask_orig[u]
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if v > u:
                        if (u, v) not in edge_set:
                            return False
                    m ^= lsb
            return True

        best_cliques = None
        best_count = float('inf')

        # Deterministic min-common run first
        det_deadline = start_time + time_budget * 0.30
        det_cliques = greedy_partition(adj_mask_orig, n, rng=None, deadline=det_deadline, mode=0)
        if validate(det_cliques, adj_mask_orig, n):
            best_cliques = det_cliques
            best_count = len(det_cliques)

        # Multi-strategy randomized restarts
        remaining = time_budget - (time.time() - start_time)
        if remaining > 3:
            num_restarts = max(1, int(remaining / 1.2))
            rng_master = random.Random(seed + 1000)
            modes = [0, 0, 0, 1, 2]  # weight toward min-common, include max + random
            for r in range(num_restarts):
                if time.time() > start_time + time_budget - 1:
                    break
                rng = random.Random(rng_master.randint(0, 2 ** 31))
                mode = modes[r % len(modes)]
                restart_deadline = min(start_time + time_budget - 1,
                                       time.time() + max(0.5, remaining / (num_restarts - r)))
                try:
                    cliques = greedy_partition(adj_mask_orig, n, rng=rng,
                                               deadline=restart_deadline, mode=mode)
                    if validate(cliques, adj_mask_orig, n):
                        if len(cliques) < best_count:
                            best_count = len(cliques)
                            best_cliques = cliques
                except Exception:
                    continue

        if best_cliques is None:
            best_cliques = []
            for u in range(n):
                m = adj_mask_orig[u]
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if v > u:
                        best_cliques.append([u, v])
                    m ^= lsb

        return best_cliques

    results = []
    # Instance 1
    t1 = min(50, TOTAL_BUDGET - 60)
    results.append(solve_instance(32, 0.6, 601, t1))

    # Instance 2
    elapsed = time.time() - start_time
    t2 = max(10, TOTAL_BUDGET - elapsed - 2)
    results.append(solve_instance(64, 0.55, 602, t2))

    return results

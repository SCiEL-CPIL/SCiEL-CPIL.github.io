import time
import random

def construct():
    start_time = time.time()
    TOTAL_BUDGET = 145  # safety margin from 150s

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

        def find_best_edge(adj, n, mode='max_common', rng=None):
            """Find edge based on selection mode.
            mode='max_common': pick edge with max common neighbors (densest region)
            mode='min_common': pick edge with min common neighbors (sparsest)
            mode='ratio': pick edge with max common/(deg_u+deg_v)
            """
            best_u, best_v = -1, -1
            best_score = None
            edges = []
            for u in range(n):
                if adj[u] == 0:
                    continue
                deg_u = adj[u].bit_count()
                for v in iter_bits(adj[u]):
                    if v > u:
                        common = (adj[u] & adj[v]).bit_count()
                        deg_v = adj[v].bit_count()
                        if mode == 'max_common':
                            score = common
                        elif mode == 'min_common':
                            score = -common  # minimize common
                        elif mode == 'ratio':
                            score = common / max(1, deg_u + deg_v - common)
                        else:
                            score = common
                        edges.append((score, u, v))

            if not edges:
                return -1, -1, float('inf')

            if rng is not None:
                # Pick from top K by score
                edges.sort(key=lambda x: x[0], reverse=(mode != 'min_common'))
                k = min(len(edges), max(3, len(edges) // 3))
                candidates = edges[:k]
                idx = rng.randint(0, len(candidates) - 1)
                best_score, best_u, best_v = candidates[idx]
            else:
                if mode == 'min_common':
                    for score, u, v in edges:
                        if best_score is None or score < best_score:
                            best_score = score
                            best_u, best_v = u, v
                else:
                    for score, u, v in edges:
                        if best_score is None or score > best_score:
                            best_score = score
                            best_u, best_v = u, v

            return best_u, best_v, best_score if best_score is not None else 0

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

        def try_merge(cliques, adj, n, deadline):
            changed = True
            while changed and (time.time() < deadline):
                changed = False
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
                for v in iter_bits(adj_mask_orig[u]):
                    if v > u:
                        if (u, v) not in edge_set:
                            return False
            return True

        def greedy_partition(adj_mask_orig, n, rng=None, deadline=None, mode='max_common'):
            if deadline is None:
                deadline = time.time() + 10
            adj = list(adj_mask_orig)
            cliques = []

            while True:
                if time.time() > deadline:
                    break
                u, v, score = find_best_edge(adj, n, mode=mode, rng=rng)
                if u == -1:
                    break
                C = expand_clique(u, v, adj, rng)
                cliques.append(C)
                remove_clique_edges(C, adj)

            cliques = try_merge(cliques, adj, n, deadline)
            return cliques

        best_cliques = None
        best_count = float('inf')

        # Try multiple modes and random restarts
        modes = ['max_common', 'min_common', 'ratio']
        
        # Deterministic runs for each mode
        for mode in modes:
            det_deadline = min(start_time + time_budget * 0.3, time.time() + 5)
            if time.time() > start_time + time_budget - 5:
                break
            det_cliques = greedy_partition(adj_mask_orig, n, rng=None, deadline=det_deadline, mode=mode)
            if validate(det_cliques, adj_mask_orig, n):
                if len(det_cliques) < best_count:
                    best_count = len(det_cliques)
                    best_cliques = det_cliques

        # Randomized restarts with best-performing mode
        remaining = time_budget - (time.time() - start_time)
        if remaining > 3:
            num_restarts = max(1, int(remaining / 1.5))
            rng_master = random.Random(seed + 1000)
            
            for r in range(num_restarts):
                if time.time() > start_time + time_budget - 1:
                    break
                rng = random.Random(rng_master.randint(0, 2**31))
                mode = modes[r % len(modes)]
                restart_deadline = min(start_time + time_budget - 1, time.time() + max(0.5, remaining / (num_restarts - r)))
                try:
                    cliques = greedy_partition(adj_mask_orig, n, rng=rng, deadline=restart_deadline, mode=mode)
                    if validate(cliques, adj_mask_orig, n):
                        if len(cliques) < best_count:
                            best_count = len(cliques)
                            best_cliques = cliques
                except:
                    continue

        if best_cliques is None:
            best_cliques = []
            for u in range(n):
                for v in iter_bits(adj_mask_orig[u]):
                    if v > u:
                        best_cliques.append([u, v])

        return best_cliques

    results = []
    # Instance 1
    t1 = min(45, TOTAL_BUDGET - 60)
    results.append(solve_instance(32, 0.6, 601, t1))

    # Instance 2
    elapsed = time.time() - start_time
    t2 = max(10, TOTAL_BUDGET - elapsed - 2)
    results.append(solve_instance(64, 0.55, 602, t2))

    return results

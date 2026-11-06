import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n = 16
    core_n = 11

    def _build_grotzsch_edges():
        edges = []
        edges.append([0, 1])
        edges.append([1, 2])
        edges.append([2, 3])
        edges.append([3, 4])
        edges.append([4, 0])
        c5_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        for u, v in c5_edges:
            edges.append([u, 5 + v])
            edges.append([v, 5 + u])
        for i in range(5):
            edges.append([10, 5 + i])
        return edges

    core_edges = _build_grotzsch_edges()

    adj_core = [0] * core_n
    for u, v in core_edges:
        adj_core[u] |= (1 << v)
        adj_core[v] |= (1 << u)

    independent_sets = []
    for mask in range(1 << core_n):
        is_ind = True
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            if adj_core[u] & mask:
                is_ind = False
                break
            m ^= lsb
        if is_ind and mask != 0:
            vertices = []
            m = mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                vertices.append(u)
                m ^= lsb
            independent_sets.append(vertices)

    independent_sets.sort(key=lambda x: -len(x))

    K = 50
    top_sets = independent_sets[:K]
    top_sets.append([])
    K_total = len(top_sets)

    D = [[False] * K_total for _ in range(K_total)]
    for i in range(K_total):
        si = set(top_sets[i])
        for j in range(K_total):
            sj = set(top_sets[j])
            D[i][j] = si.isdisjoint(sj)

    pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

    def _is_tf_mask(mask):
        adj_new = [0] * 5
        for k in range(10):
            if mask & (1 << k):
                u, v = pairs[k]
                adj_new[u] |= (1 << v)
                adj_new[v] |= (1 << u)
        for u in range(5):
            for v in range(u + 1, 5):
                if adj_new[u] & (1 << v):
                    if adj_new[u] & adj_new[v]:
                        return False
        return True

    max_tf_lookup = [0] * (1 << 10)
    for mask in range(1 << 10):
        if _is_tf_mask(mask):
            max_tf_lookup[mask] = bin(mask).count("1")
        else:
            best = 0
            sub = mask
            while sub:
                if _is_tf_mask(sub):
                    cnt = bin(sub).count("1")
                    if cnt > best:
                        best = cnt
                sub = (sub - 1) & mask
            max_tf_lookup[mask] = best

    def _is_3_colorable(adj, nn):
        degrees = [bin(adj[i]).count("1") for i in range(nn)]
        order = sorted(range(nn), key=lambda x: -degrees[x])
        color = [-1] * nn

        def backtrack(idx):
            if idx == nn:
                return True
            v = order[idx]
            used = 0
            mask = adj[v]
            m = mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                if color[u] != -1:
                    used |= (1 << color[u])
                m ^= lsb
            for c in range(3):
                if not (used & (1 << c)):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False

        return backtrack(0)

    def _evaluate(indices):
        adj = [0] * n
        for u, v in core_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        edge_count = len(core_edges)
        new_vertices = list(range(core_n, n))

        for pos in range(5):
            v_new = new_vertices[pos]
            s = top_sets[indices[pos]]
            for v_core in s:
                adj[v_new] |= (1 << v_core)
                adj[v_core] |= (1 << v_new)
                edge_count += 1

        possible_mask = 0
        for k in range(10):
            i, j = pairs[k]
            if D[indices[i]][indices[j]]:
                possible_mask |= (1 << k)

        best_count = max_tf_lookup[possible_mask]
        edge_count += best_count

        if _is_3_colorable(adj, n):
            return (edge_count, False)
        else:
            return (edge_count, True)

    rng = random.Random(42)
    best_score = -1
    best_indices = None

    for restart in range(200):
        if time.time() > deadline:
            break

        indices = [rng.randint(0, K_total - 1) for _ in range(5)]
        score, valid = _evaluate(indices)
        if valid:
            current_score = score
        else:
            current_score = -1

        if current_score > best_score:
            best_score = current_score
            best_indices = list(indices)

        if current_score < 0:
            continue

        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(5):
                for new_idx in range(K_total):
                    if new_idx == indices[i]:
                        continue
                    old_idx = indices[i]
                    indices[i] = new_idx
                    new_score, new_valid = _evaluate(indices)
                    if new_valid and new_score > current_score:
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_indices = list(indices)
                    else:
                        indices[i] = old_idx

    # Greedy triangle-free edge expansion (from inspirations)
    def _greedy_expand(adj, seed):
        possible = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    possible.append((u, v))
        random.Random(seed).shuffle(possible)
        added = 0
        for u, v in possible:
            if time.time() > deadline:
                break
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
        return added

    def _edges_from_adj(adj):
        edges = []
        for u in range(n):
            m = adj[u] & ~((1 << (u + 1)) - 1)
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                edges.append([u, v])
                m ^= lsb
        return edges

    def _is_triangle_free_adj(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    if adj[u] & adj[v]:
                        return False
        return True

    def _build_base_adj(indices):
        adj = [0] * n
        for u, v in core_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        new_vertices = list(range(core_n, n))
        for pos in range(5):
            v_new = new_vertices[pos]
            s = top_sets[indices[pos]]
            for v_core in s:
                adj[v_new] |= (1 << v_core)
                adj[v_core] |= (1 << v_new)
        i0, i1, i2, i3, i4 = indices
        mask = 0
        if D[i0][i1]: mask |= 1 << 0
        if D[i0][i2]: mask |= 1 << 1
        if D[i0][i3]: mask |= 1 << 2
        if D[i0][i4]: mask |= 1 << 3
        if D[i1][i2]: mask |= 1 << 4
        if D[i1][i3]: mask |= 1 << 5
        if D[i1][i4]: mask |= 1 << 6
        if D[i2][i3]: mask |= 1 << 7
        if D[i2][i4]: mask |= 1 << 8
        if D[i3][i4]: mask |= 1 << 9
        best_sub_mask = 0
        best_count = 0
        for sub_mask in range(1024):
            if sub_mask & ~mask != 0:
                continue
            if max_tf_lookup[sub_mask] > best_count:
                best_count = max_tf_lookup[sub_mask]
                best_sub_mask = sub_mask
        for k in range(10):
            if best_sub_mask & (1 << k):
                u, v = pairs[k]
                adj[new_vertices[u]] |= (1 << new_vertices[v])
                adj[new_vertices[v]] |= (1 << new_vertices[u])
        return adj

    def _is_4chromatic_adj(adj):
        return not _is_3_colorable(adj, n)

    # Fallback: known-valid Grötzsch-based graph (never invalid)
    def _fallback_edges():
        return list(core_edges)

    candidate_edges = None

    # Primary candidate: greedy-expand the best local-search structure
    if best_indices is not None and time.time() < deadline:
        base_adj = _build_base_adj(best_indices)
        if _is_4chromatic_adj(base_adj):
            # expand from multiple random seeds, keep best valid
            local_best = _edges_from_adj(base_adj)
            local_best_count = len(local_best)
            for seed in range(20):
                if time.time() > deadline:
                    break
                exp_adj = list(base_adj)
                _greedy_expand(exp_adj, seed)
                if _is_triangle_free_adj(exp_adj) and _is_4chromatic_adj(exp_adj):
                    cand = _edges_from_adj(exp_adj)
                    if len(cand) > local_best_count:
                        local_best_count = len(cand)
                        local_best = cand
            candidate_edges = local_best
            best_count = local_best_count
        else:
            candidate_edges = _edges_from_adj(base_adj)
            best_count = len(candidate_edges)
    else:
        candidate_edges = _fallback_edges()
        best_count = len(candidate_edges)

    # Secondary: independent greedy-expansion attempts (inspiration 1/2 strategy)
    for attempt in range(60):
        if time.time() > deadline:
            break
        if attempt % 2 == 0:
            base_adj = _build_base_adj([0, 0, 0, 0, 0])
        else:
            base_adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        if (base_adj[u] & base_adj[v]) == 0:
                            base_adj[u] |= (1 << v)
                            base_adj[v] |= (1 << u)
        if _is_4chromatic_adj(base_adj):
            _greedy_expand(base_adj, 1000 + attempt)
            if _is_triangle_free_adj(base_adj) and _is_4chromatic_adj(base_adj):
                cand = _edges_from_adj(base_adj)
                if len(cand) > best_count:
                    best_count = len(cand)
                    candidate_edges = cand

    # Final safety validation: ensure triangle-free; fall back if not
    if candidate_edges is None:
        candidate_edges = _fallback_edges()
    else:
        check_adj = [0] * n
        for u, v in candidate_edges:
            check_adj[u] |= (1 << v)
            check_adj[v] |= (1 << u)
        if not _is_triangle_free_adj(check_adj):
            candidate_edges = _fallback_edges()

    return candidate_edges

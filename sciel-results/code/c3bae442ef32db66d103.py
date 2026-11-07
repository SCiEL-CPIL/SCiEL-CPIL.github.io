def sign_edges(graph, budget):
    import numpy as np
    import time

    start_time = time.time()
    DEADLINE = 58

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m

    edge_u = np.array([u for u, v in edges], dtype=int)
    edge_v = np.array([v for u, v in edges], dtype=int)

    # Build spanning tree for fundamental cycle checks
    adj_list = [[] for _ in range(n_vertices)]
    for i in range(n_edges):
        u, v = edge_u[i], edge_v[i]
        adj_list[u].append((v, i))
        adj_list[v].append((u, i))

    parent = [-1] * n_vertices
    parent_edge = [-1] * n_vertices
    depth = [0] * n_vertices
    visited = [False] * n_vertices
    visited[0] = True
    queue = [0]
    qi = 0
    while qi < len(queue):
        node = queue[qi]
        qi += 1
        for nb, ei in adj_list[node]:
            if not visited[nb]:
                visited[nb] = True
                parent[nb] = node
                parent_edge[nb] = ei
                depth[nb] = depth[node] + 1
                queue.append(nb)

    tree_edges = set(parent_edge[1:])
    non_tree_edges = [i for i in range(n_edges) if i not in tree_edges]

    def get_lca(u, v):
        while depth[u] > depth[v]:
            u = parent[u]
        while depth[v] > depth[u]:
            v = parent[v]
        while u != v:
            u = parent[u]
            v = parent[v]
        return u

    fundamental_cycles = {}
    for i in non_tree_edges:
        u, v = edge_u[i], edge_v[i]
        lca = get_lca(u, v)
        cycle_edges = set()
        cycle_edges.add(i)
        cur = u
        while cur != lca:
            cycle_edges.add(parent_edge[cur])
            cur = parent[cur]
        cur = v
        while cur != lca:
            cycle_edges.add(parent_edge[cur])
            cur = parent[cur]
        fundamental_cycles[i] = cycle_edges

    def is_valid_lift_fast(signs):
        """Lift is connected iff signed graph is NOT balanced (some cycle has odd # of -1s)."""
        for i in non_tree_edges:
            product = 1
            for ei in fundamental_cycles[i]:
                product *= signs[ei]
            if product == -1:
                return True
        return False

    def build_signed_adj(signs):
        A = np.zeros((n_vertices, n_vertices))
        for i in range(n_edges):
            s = signs[i]
            u = edge_u[i]
            v = edge_v[i]
            A[u, v] = s
            A[v, u] = s
        return A

    def spectral_radius_from_A(A):
        return np.max(np.abs(np.linalg.eigvalsh(A)))

    def generate_initial_patterns(rng):
        patterns = []
        patterns.append(np.ones(n_edges, dtype=int))
        patterns.append(np.full(n_edges, -1, dtype=int))
        patterns.append(rng.choice([-1, 1], size=n_edges).astype(int))
        s = np.ones(n_edges, dtype=int)
        for i, (u, v) in enumerate(edges):
            if u < m and v < m:
                s[i] = 1 if (min(u, v) % 2 == 0) else -1
        patterns.append(s)
        s = np.ones(n_edges, dtype=int)
        for i, (u, v) in enumerate(edges):
            if not (u < m and v < m) and not (u >= m and v >= m):
                s[i] = -1
        patterns.append(s)
        s = np.ones(n_edges, dtype=int)
        for i, (u, v) in enumerate(edges):
            if u < m and v < m:
                s[i] = -1
        patterns.append(s)
        s = np.ones(n_edges, dtype=int)
        for i, (u, v) in enumerate(edges):
            if u >= m and v >= m:
                s[i] = -1
        patterns.append(s)
        patterns.append(np.where(rng.random(n_edges) < 0.6, -1, 1).astype(int))
        patterns.append(rng.choice([-1, 1], size=n_edges).astype(int))
        patterns.append(np.where(rng.random(n_edges) < 0.7, -1, 1).astype(int))
        return patterns

    def make_valid(signs, rng):
        if is_valid_lift_fast(signs):
            return signs
        for _ in range(30):
            idx = rng.integers(0, n_edges)
            signs[idx] *= -1
            if is_valid_lift_fast(signs):
                return signs
        for i in range(n_edges):
            signs[i] *= -1
            if is_valid_lift_fast(signs):
                return signs
            signs[i] *= -1
        signs = np.ones(n_edges, dtype=int)
        for i in range(n_edges):
            u, v = edge_u[i], edge_v[i]
            if not (u < m and v < m) and not (u >= m and v >= m):
                signs[i] = -1
                break
        return signs

    best_signs = None
    best_radius = float('inf')
    rng = np.random.default_rng(3)

    for init_signs in generate_initial_patterns(rng):
        if time.time() - start_time > DEADLINE - 3:
            break

        signs = init_signs.copy()
        signs = make_valid(signs, rng)
        if not is_valid_lift_fast(signs):
            continue

        A = build_signed_adj(signs)
        current_radius = spectral_radius_from_A(A)

        improved = True
        while improved and (time.time() - start_time < DEADLINE - 1):
            improved = False
            for i in range(n_edges):
                if time.time() - start_time > DEADLINE - 1:
                    break
                old = signs[i]
                u = edge_u[i]
                v = edge_v[i]
                signs[i] = -old
                A[u, v] = -A[u, v]
                A[v, u] = -A[v, u]

                if not is_valid_lift_fast(signs):
                    signs[i] = old
                    A[u, v] = -A[u, v]
                    A[v, u] = -A[v, u]
                    continue

                new_radius = spectral_radius_from_A(A)
                if new_radius < current_radius - 1e-10:
                    current_radius = new_radius
                    improved = True
                else:
                    signs[i] = old
                    A[u, v] = -A[u, v]
                    A[v, u] = -A[v, u]

            if not improved:
                for _ in range(30):
                    if time.time() - start_time > DEADLINE - 1:
                        break
                    i, j = rng.choice(n_edges, size=2, replace=False)
                    ui, vi = edge_u[i], edge_v[i]
                    uj, vj = edge_u[j], edge_v[j]
                    oi, oj = signs[i], signs[j]
                    signs[i] = -oi
                    signs[j] = -oj
                    A[ui, vi] = -A[ui, vi]
                    A[vi, ui] = -A[vi, ui]
                    A[uj, vj] = -A[uj, vj]
                    A[vj, uj] = -A[vj, uj]

                    if not is_valid_lift_fast(signs):
                        signs[i] = oi
                        signs[j] = oj
                        A[ui, vi] = -A[ui, vi]
                        A[vi, ui] = -A[vi, ui]
                        A[uj, vj] = -A[uj, vj]
                        A[vj, uj] = -A[vj, uj]
                        continue

                    new_radius = spectral_radius_from_A(A)
                    if new_radius < current_radius - 1e-10:
                        current_radius = new_radius
                        improved = True
                    else:
                        signs[i] = oi
                        signs[j] = oj
                        A[ui, vi] = -A[ui, vi]
                        A[vi, ui] = -A[vi, ui]
                        A[uj, vj] = -A[uj, vj]
                        A[vj, uj] = -A[vj, uj]

        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = signs.copy()

    if best_signs is not None and time.time() - start_time < DEADLINE - 2:
        current_signs = best_signs.copy()
        A = build_signed_adj(current_signs)
        current_radius = best_radius

        for _ in range(500):
            if time.time() - start_time > DEADLINE - 1:
                break

            n_flips = rng.integers(1, 4)
            idxs = rng.choice(n_edges, size=n_flips, replace=False)
            old_signs = current_signs[idxs].copy()
            for idx in idxs:
                current_signs[idx] *= -1
                u, v = edge_u[idx], edge_v[idx]
                A[u, v] = -A[u, v]
                A[v, u] = -A[v, u]

            if not is_valid_lift_fast(current_signs):
                for k, idx in enumerate(idxs):
                    current_signs[idx] = old_signs[k]
                    u, v = edge_u[idx], edge_v[idx]
                    A[u, v] = -A[u, v]
                    A[v, u] = -A[v, u]
                continue

            new_radius = spectral_radius_from_A(A)

            improved = True
            while improved and (time.time() - start_time < DEADLINE - 1):
                improved = False
                for i in range(n_edges):
                    if time.time() - start_time > DEADLINE - 1:
                        break
                    old = current_signs[i]
                    u, v = edge_u[i], edge_v[i]
                    current_signs[i] = -old
                    A[u, v] = -A[u, v]
                    A[v, u] = -A[v, u]

                    if not is_valid_lift_fast(current_signs):
                        current_signs[i] = old
                        A[u, v] = -A[u, v]
                        A[v, u] = -A[v, u]
                        continue

                    r = spectral_radius_from_A(A)
                    if r < new_radius - 1e-10:
                        new_radius = r
                        improved = True
                    else:
                        current_signs[i] = old
                        A[u, v] = -A[u, v]
                        A[v, u] = -A[v, u]

            if new_radius < best_radius:
                best_radius = new_radius
                best_signs = current_signs.copy()
                current_signs = best_signs.copy()
                A = build_signed_adj(current_signs)
            else:
                current_signs = best_signs.copy()
                A = build_signed_adj(current_signs)

    if best_signs is None:
        best_signs = np.ones(n_edges, dtype=int)
        for i in range(n_edges):
            u, v = edge_u[i], edge_v[i]
            if not (u < m and v < m) and not (u >= m and v >= m):
                best_signs[i] = -1
                break
        if not is_valid_lift_fast(best_signs):
            best_signs = make_valid(best_signs, rng)

    return best_signs

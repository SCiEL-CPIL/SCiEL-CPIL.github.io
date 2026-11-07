def sign_edges(graph, budget):
    import numpy as np
    import time
    from scipy.linalg import eigh

    start_time = time.time()
    DEADLINE = 55  # safety margin before 60s

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
        """Compute spectral radius using scipy for speed."""
        try:
            vals = eigh(A, eigvals_only=True)
            return np.max(np.abs(vals))
        except:
            return np.max(np.abs(np.linalg.eigvalsh(A)))

    def make_valid(signs, rng):
        if is_valid_lift_fast(signs):
            return signs
        for _ in range(50):
            idx = rng.integers(0, n_edges)
            signs[idx] *= -1
            if is_valid_lift_fast(signs):
                return signs
        for i in range(n_edges):
            signs[i] *= -1
            if is_valid_lift_fast(signs):
                return signs
            signs[i] *= -1
        # Fallback: flip one matching edge
        for i in range(n_edges):
            u, v = edge_u[i], edge_v[i]
            if not (u < m and v < m) and not (u >= m and v >= m):
                signs[i] = -1
                break
        return signs

    def local_search(signs, rng, max_time):
        """Single-edge and multi-edge flip local search."""
        A = build_signed_adj(signs)
        current_radius = spectral_radius_from_A(A)
        improved = True
        while improved and (time.time() - start_time < max_time):
            improved = False
            # Try single edge flips
            for i in range(n_edges):
                if time.time() - start_time > max_time:
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
                if new_radius < current_radius - 1e-12:
                    current_radius = new_radius
                    improved = True
                else:
                    signs[i] = old
                    A[u, v] = -A[u, v]
                    A[v, u] = -A[v, u]

            # Try double edge flips
            if not improved:
                for _ in range(20):
                    if time.time() - start_time > max_time:
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
                    if new_radius < current_radius - 1e-12:
                        current_radius = new_radius
                        improved = True
                    else:
                        signs[i] = oi
                        signs[j] = oj
                        A[ui, vi] = -A[ui, vi]
                        A[vi, ui] = -A[vi, ui]
                        A[uj, vj] = -A[uj, vj]
                        A[vj, uj] = -A[vj, uj]

        return signs, current_radius

    def multi_flip_search(signs, rng, max_time):
        """Multi-edge flip search with local polish."""
        A = build_signed_adj(signs)
        current_radius = spectral_radius_from_A(A)
        best_signs = signs.copy()
        best_radius = current_radius

        for _ in range(200):
            if time.time() - start_time > max_time:
                break

            n_flips = rng.integers(1, 4)
            idxs = rng.choice(n_edges, size=n_flips, replace=False)
            old_signs = signs[idxs].copy()
            for idx in idxs:
                signs[idx] *= -1
                u, v = edge_u[idx], edge_v[idx]
                A[u, v] = -A[u, v]
                A[v, u] = -A[v, u]

            if not is_valid_lift_fast(signs):
                for k, idx in enumerate(idxs):
                    signs[idx] = old_signs[k]
                    u, v = edge_u[idx], edge_v[idx]
                    A[u, v] = -A[u, v]
                    A[v, u] = -A[v, u]
                continue

            new_radius = spectral_radius_from_A(A)

            # Local polish
            polished_signs, polished_radius = local_search(signs.copy(), rng, min(max_time, time.time() - start_time + 2))
            if polished_radius < best_radius:
                best_radius = polished_radius
                best_signs = polished_signs.copy()
                signs = polished_signs.copy()
                A = build_signed_adj(signs)
            else:
                # Revert
                for k, idx in enumerate(idxs):
                    signs[idx] = old_signs[k]
                    u, v = edge_u[idx], edge_v[idx]
                    A[u, v] = -A[u, v]
                    A[v, u] = -A[v, u]

        return best_signs, best_radius

    # Generate diverse initial patterns
    def generate_initial_patterns(rng):
        patterns = []
        
        # All +1
        patterns.append(np.ones(n_edges, dtype=int))
        
        # All -1
        patterns.append(np.full(n_edges, -1, dtype=int))
        
        # Random
        patterns.append(rng.choice([-1, 1], size=n_edges).astype(int))
        
        # Alternating on bottom cycle
        s = np.ones(n_edges, dtype=int)
        for i, (u, v) in enumerate(edges):
            if u < m and v < m:
                s[i] = 1 if (min(u, v) % 2 == 0) else -1
        patterns.append(s)
        
        # Alternating on top cycle
        s = np.ones(n_edges, dtype=int)
        for i, (u, v) in enumerate(edges):
            if u >= m and v >= m:
                s[i] = 1 if (min(u - m, v - m) % 2 == 0) else -1
        patterns.append(s)
        
        # All matching edges -1
        s = np.ones(n_edges, dtype=int)
        for i, (u, v) in enumerate(edges):
            if not (u < m and v < m) and not (u >= m and v >= m):
                s[i] = -1
        patterns.append(s)
        
        # Bottom cycle all -1
        s = np.ones(n_edges, dtype=int)
        for i, (u, v) in enumerate(edges):
            if u < m and v < m:
                s[i] = -1
        patterns.append(s)
        
        # Top cycle all -1
        s = np.ones(n_edges, dtype=int)
        for i, (u, v) in enumerate(edges):
            if u >= m and v >= m:
                s[i] = -1
        patterns.append(s)
        
        # Biased random (60% -1)
        patterns.append(np.where(rng.random(n_edges) < 0.6, -1, 1).astype(int))
        
        # Biased random (70% -1)
        patterns.append(np.where(rng.random(n_edges) < 0.7, -1, 1).astype(int))
        
        # Biased random (50% -1)
        patterns.append(np.where(rng.random(n_edges) < 0.5, -1, 1).astype(int))
        
        # Pattern: alternate signs on matching
        s = np.ones(n_edges, dtype=int)
        for i, (u, v) in enumerate(edges):
            if not (u < m and v < m) and not (u >= m and v >= m):
                idx = min(u, v)
                s[i] = 1 if (idx % 2 == 0) else -1
        patterns.append(s)
        
        # Pattern: checkerboard on both cycles
        s = np.ones(n_edges, dtype=int)
        for i, (u, v) in enumerate(edges):
            if u < m and v < m:
                s[i] = 1 if (u % 2 == v % 2) else -1
            elif u >= m and v >= m:
                s[i] = 1 if ((u-m) % 2 == (v-m) % 2) else -1
        patterns.append(s)
        
        return patterns

    best_signs = None
    best_radius = float('inf')
    rng = np.random.default_rng(3)

    # Phase 1: Local search from diverse initial patterns
    for init_signs in generate_initial_patterns(rng):
        if time.time() - start_time > DEADLINE - 10:
            break

        signs = init_signs.copy()
        signs = make_valid(signs, rng)
        if not is_valid_lift_fast(signs):
            continue

        polished_signs, current_radius = local_search(signs, rng, DEADLINE - 5)
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = polished_signs.copy()

    # Phase 2: Multi-flip search from best found
    if best_signs is not None and time.time() - start_time < DEADLINE - 3:
        refined_signs, refined_radius = multi_flip_search(best_signs.copy(), rng, DEADLINE - 1)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_signs = refined_signs.copy()

    # Phase 3: Final polish
    if best_signs is not None and time.time() - start_time < DEADLINE - 1:
        final_signs, final_radius = local_search(best_signs.copy(), rng, DEADLINE)
        if final_radius < best_radius:
            best_radius = final_radius
            best_signs = final_signs.copy()

    if best_signs is None:
        # Fallback valid sign assignment
        best_signs = np.ones(n_edges, dtype=int)
        for i in range(n_edges):
            u, v = edge_u[i], edge_v[i]
            if not (u < m and v < m) and not (u >= m and v >= m):
                best_signs[i] = -1
                break
        if not is_valid_lift_fast(best_signs):
            best_signs = make_valid(best_signs, rng)

    return best_signs

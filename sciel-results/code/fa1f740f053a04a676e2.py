def sign_edges(graph, budget):
    import numpy as np
    import time
    from collections import deque

    start_time = time.time()
    DEADLINE = 58  # Safety margin before 60s limit

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m
    n_lift = 2 * n_vertices

    # Classify edges: cycle1 (0..m-1), cycle2 (m..2m-1), vertical
    edge_types = []
    for (u, v) in edges:
        if u < m and v < m:
            edge_types.append(0)
        elif u >= m and v >= m:
            edge_types.append(1)
        else:
            edge_types.append(2)

    # Precompute edge endpoints as arrays for fast access
    edge_u = np.array([u for u, v in edges], dtype=int)
    edge_v = np.array([v for u, v in edges], dtype=int)

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

    def is_valid_lift_fast(signs):
        """Check if the 2-lift is connected using BFS on the lift."""
        adj = [[] for _ in range(n_lift)]
        for i in range(n_edges):
            s = signs[i]
            u = edge_u[i]
            v = edge_v[i]
            if s == 1:
                adj[2*u].append(2*v)
                adj[2*v].append(2*u)
                adj[2*u+1].append(2*v+1)
                adj[2*v+1].append(2*u+1)
            else:
                adj[2*u].append(2*v+1)
                adj[2*v+1].append(2*u)
                adj[2*u+1].append(2*v)
                adj[2*v].append(2*u+1)

        visited = [False] * n_lift
        queue = deque([0])
        visited[0] = True
        count = 1
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    queue.append(nb)
        return count == n_lift

    def make_valid(signs, rng):
        """Ensure signs produce a connected lift."""
        if is_valid_lift_fast(signs):
            return signs
        # Flip random edges until valid
        for _ in range(50):
            idx = rng.integers(0, n_edges)
            signs[idx] *= -1
            if is_valid_lift_fast(signs):
                return signs
        # Try flipping each single edge
        for i in range(n_edges):
            signs[i] *= -1
            if is_valid_lift_fast(signs):
                return signs
            signs[i] *= -1
        # Fallback: known valid pattern - all +1 except one vertical edge = -1
        signs = np.ones(n_edges, dtype=int)
        for i, t in enumerate(edge_types):
            if t == 2:
                signs[i] = -1
                break
        return signs

    # Generate diverse initial patterns
    def generate_initial_patterns(rng):
        patterns = []

        # 1. All +1
        s = np.ones(n_edges, dtype=int)
        patterns.append(s.copy())

        # 2. All -1
        s = np.full(n_edges, -1, dtype=int)
        patterns.append(s.copy())

        # 3. Random
        patterns.append(rng.choice([-1, 1], size=n_edges).astype(int))

        # 4. Alternate on cycle 1
        s = np.ones(n_edges, dtype=int)
        for i, (u, v) in enumerate(edges):
            if u < m and v < m:
                s[i] = 1 if (min(u, v) % 2 == 0) else -1
        patterns.append(s.copy())

        # 5. All -1 on verticals
        s = np.ones(n_edges, dtype=int)
        for i, t in enumerate(edge_types):
            if t == 2:
                s[i] = -1
        patterns.append(s.copy())

        # 6. All -1 on cycle 1
        s = np.ones(n_edges, dtype=int)
        for i, t in enumerate(edge_types):
            if t == 0:
                s[i] = -1
        patterns.append(s.copy())

        # 7. All -1 on cycle 2
        s = np.ones(n_edges, dtype=int)
        for i, t in enumerate(edge_types):
            if t == 1:
                s[i] = -1
        patterns.append(s.copy())

        # 8. Checkerboard-like pattern
        s = np.ones(n_edges, dtype=int)
        for i, (u, v) in enumerate(edges):
            s[i] = 1 if ((u + v) % 2 == 0) else -1
        patterns.append(s.copy())

        # 9. Random with bias toward -1
        s = np.where(rng.random(n_edges) < 0.6, -1, 1).astype(int)
        patterns.append(s.copy())

        # 10. Another random
        patterns.append(rng.choice([-1, 1], size=n_edges).astype(int))

        # 11. Pattern: -1 on every other vertical
        s = np.ones(n_edges, dtype=int)
        vert_count = 0
        for i, t in enumerate(edge_types):
            if t == 2:
                if vert_count % 2 == 0:
                    s[i] = -1
                vert_count += 1
        patterns.append(s.copy())

        # 12. Balanced: half -1, half +1, alternating by index
        s = np.ones(n_edges, dtype=int)
        for i in range(n_edges):
            if i % 2 == 0:
                s[i] = -1
        patterns.append(s.copy())

        # 13. Cycle 1 all -1, cycle 2 all +1, verticals alternating
        s = np.ones(n_edges, dtype=int)
        vert_count = 0
        for i, t in enumerate(edge_types):
            if t == 0:
                s[i] = -1
            elif t == 2:
                if vert_count % 2 == 0:
                    s[i] = -1
                vert_count += 1
        patterns.append(s.copy())

        # 14. Cycle 1 alternating, cycle 2 alternating (opposite phase)
        s = np.ones(n_edges, dtype=int)
        for i, (u, v) in enumerate(edges):
            if u < m and v < m:
                s[i] = 1 if (min(u, v) % 2 == 0) else -1
            elif u >= m and v >= m:
                s[i] = -1 if (min(u-m, v-m) % 2 == 0) else 1
        patterns.append(s.copy())

        # 15. High-frustration: many -1s
        s = np.where(rng.random(n_edges) < 0.7, -1, 1).astype(int)
        patterns.append(s.copy())

        return patterns

    best_signs = None
    best_radius = float('inf')

    rng = np.random.default_rng(3)

    # Phase 1: Evaluate diverse initial patterns with local search
    initial_patterns = generate_initial_patterns(rng)

    for init_signs in initial_patterns:
        if time.time() - start_time > DEADLINE - 3:
            break

        signs = init_signs.copy()
        signs = make_valid(signs, rng)

        if not is_valid_lift_fast(signs):
            continue

        A = build_signed_adj(signs)
        current_radius = spectral_radius_from_A(A)

        # Local search: 1-flip
        improved = True
        while improved and (time.time() - start_time < DEADLINE - 1):
            improved = False
            # Try all 1-flips
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

            # If no 1-flip helped, try random 2-flips
            if not improved:
                for _ in range(30):
                    if time.time() - start_time > DEADLINE - 1:
                        break
                    i, j = rng.choice(n_edges, size=2, replace=False)
                    ui = edge_u[i]
                    vi = edge_v[i]
                    uj = edge_u[j]
                    vj = edge_v[j]
                    oi = signs[i]
                    oj = signs[j]
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

    # Phase 2: More aggressive search from best with perturbation
    if best_signs is not None and time.time() - start_time < DEADLINE - 2:
        current_signs = best_signs.copy()
        A = build_signed_adj(current_signs)
        current_radius = best_radius

        # Iterative perturbation and local search
        for _ in range(300):
            if time.time() - start_time > DEADLINE - 1:
                break

            # Perturb: flip 1-3 random edges
            n_flips = rng.integers(1, 4)
            idxs = rng.choice(n_edges, size=n_flips, replace=False)
            old_signs = current_signs[idxs].copy()
            for idx in idxs:
                current_signs[idx] *= -1
                u = edge_u[idx]
                v = edge_v[idx]
                A[u, v] = -A[u, v]
                A[v, u] = -A[v, u]

            if not is_valid_lift_fast(current_signs):
                for k, idx in enumerate(idxs):
                    current_signs[idx] = old_signs[k]
                    u = edge_u[idx]
                    v = edge_v[idx]
                    A[u, v] = -A[u, v]
                    A[v, u] = -A[v, u]
                continue

            new_radius = spectral_radius_from_A(A)

            # Local polish from perturbed state
            improved = True
            while improved and (time.time() - start_time < DEADLINE - 1):
                improved = False
                for i in range(n_edges):
                    if time.time() - start_time > DEADLINE - 1:
                        break
                    old = current_signs[i]
                    u = edge_u[i]
                    v = edge_v[i]
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
                # Reset current to best for next perturbation
                current_signs = best_signs.copy()
                A = build_signed_adj(current_signs)
            else:
                # Revert to best
                current_signs = best_signs.copy()
                A = build_signed_adj(current_signs)

    # Phase 3: Final high-precision polish
    if best_signs is not None and time.time() - start_time < DEADLINE - 0.5:
        current_signs = best_signs.copy()
        A = build_signed_adj(current_signs)
        current_radius = best_radius

        # Tighter local search with smaller tolerance
        for _ in range(5):
            improved = True
            while improved and (time.time() - start_time < DEADLINE - 0.5):
                improved = False
                for i in range(n_edges):
                    if time.time() - start_time > DEADLINE - 0.5:
                        break
                    old = current_signs[i]
                    u = edge_u[i]
                    v = edge_v[i]
                    current_signs[i] = -old
                    A[u, v] = -A[u, v]
                    A[v, u] = -A[v, u]

                    if not is_valid_lift_fast(current_signs):
                        current_signs[i] = old
                        A[u, v] = -A[u, v]
                        A[v, u] = -A[v, u]
                        continue

                    r = spectral_radius_from_A(A)
                    if r < current_radius - 1e-12:
                        current_radius = r
                        improved = True
                    else:
                        current_signs[i] = old
                        A[u, v] = -A[u, v]
                        A[v, u] = -A[v, u]

        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = current_signs.copy()

    if best_signs is None:
        # Ultimate fallback
        best_signs = np.ones(n_edges, dtype=int)
        for i, t in enumerate(edge_types):
            if t == 2:
                best_signs[i] = -1
                break
        if not is_valid_lift_fast(best_signs):
            best_signs = make_valid(best_signs, rng)

    return best_signs

def sign_edges(graph, budget):
    import numpy as np
    import time
    from scipy.linalg import eigh

    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m

    edge_u = np.array([u for u, v in edges], dtype=int)
    edge_v = np.array([v for u, v in edges], dtype=int)

    # Build spanning tree via BFS
    adj_list = [[] for _ in range(n_vertices)]
    for i in range(n_edges):
        u, v = edge_u[i], edge_v[i]
        adj_list[u].append((v, i))
        adj_list[v].append((u, i))

    parent = [-1] * n_vertices
    parent_edge = [-1] * n_vertices
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
                queue.append(nb)

    tree_edges = set(parent_edge[1:])
    non_tree_edges = [i for i in range(n_edges) if i not in tree_edges]
    n_non_tree = len(non_tree_edges)  # should be m+1

    def build_A_from_non_tree_signs(non_tree_signs):
        A = np.zeros((n_vertices, n_vertices))
        for v in range(1, n_vertices):
            pe = parent_edge[v]
            u = parent[v]
            A[u, v] = 1
            A[v, u] = 1
        for k, ei in enumerate(non_tree_edges):
            s = non_tree_signs[k]
            u = edge_u[ei]
            v = edge_v[ei]
            A[u, v] = s
            A[v, u] = s
        return A

    def spectral_radius(A):
        try:
            vals = eigh(A, eigvals_only=True)
            return np.max(np.abs(vals))
        except:
            return np.max(np.abs(np.linalg.eigvalsh(A)))

    def is_connected_lift(A):
        for ei in non_tree_edges:
            if A[edge_u[ei], edge_v[ei]] == -1:
                return True
        return False

    # Enumerate all 2^(m+1) switching classes
    best_radius = float('inf')
    best_non_tree_signs = None

    n_combinations = 1 << n_non_tree
    for mask in range(n_combinations):
        if time.time() - start_time > DEADLINE - 30:
            break

        non_tree_signs = np.ones(n_non_tree, dtype=int)
        for k in range(n_non_tree):
            if mask & (1 << k):
                non_tree_signs[k] = -1

        A = build_A_from_non_tree_signs(non_tree_signs)

        if not is_connected_lift(A):
            continue

        radius = spectral_radius(A)
        if radius < best_radius:
            best_radius = radius
            best_non_tree_signs = non_tree_signs.copy()

    # Convert best non-tree signs to full sign vector
    if best_non_tree_signs is not None:
        signs = np.ones(n_edges, dtype=int)
        for v in range(1, n_vertices):
            pe = parent_edge[v]
            signs[pe] = 1
        for k, ei in enumerate(non_tree_edges):
            signs[ei] = best_non_tree_signs[k]
    else:
        signs = np.ones(n_edges, dtype=int)
        for ei in non_tree_edges:
            signs[ei] = -1
            break

    # Build initial A
    A = np.zeros((n_vertices, n_vertices))
    for i in range(n_edges):
        s = signs[i]
        u = edge_u[i]
        v = edge_v[i]
        A[u, v] = s
        A[v, u] = s

    current_radius = spectral_radius(A)

    # Local search: flip individual edges
    def local_search(signs, A, current_radius, max_time):
        improved = True
        while improved and (time.time() - start_time < max_time):
            improved = False
            for i in range(n_edges):
                if time.time() - start_time > max_time:
                    break
                old = signs[i]
                u = edge_u[i]
                v = edge_v[i]
                signs[i] = -old
                A[u, v] = -A[u, v]
                A[v, u] = -A[v, u]

                if not is_connected_lift(A):
                    signs[i] = old
                    A[u, v] = -A[u, v]
                    A[v, u] = -A[v, u]
                    continue

                new_radius = spectral_radius(A)
                if new_radius < current_radius - 1e-12:
                    current_radius = new_radius
                    improved = True
                else:
                    signs[i] = old
                    A[u, v] = -A[u, v]
                    A[v, u] = -A[v, u]
        return signs, current_radius

    # First local search pass
    if time.time() - start_time < DEADLINE - 10:
        signs, current_radius = local_search(signs.copy(), A.copy(), current_radius, DEADLINE - 5)

    # Random restarts with local search
    rng = np.random.default_rng(42)
    for _ in range(50):
        if time.time() - start_time > DEADLINE - 5:
            break
        trial_signs = signs.copy()
        n_flips = rng.integers(1, 5)
        for _ in range(n_flips):
            idx = rng.integers(0, n_edges)
            trial_signs[idx] *= -1
        trial_A = np.zeros((n_vertices, n_vertices))
        for i in range(n_edges):
            s = trial_signs[i]
            u = edge_u[i]
            v = edge_v[i]
            trial_A[u, v] = s
            trial_A[v, u] = s
        if not is_connected_lift(trial_A):
            continue
        trial_signs, trial_radius = local_search(trial_signs, trial_A, spectral_radius(trial_A), DEADLINE - 2)
        if trial_radius < current_radius:
            current_radius = trial_radius
            signs = trial_signs
            A = trial_A

    return signs

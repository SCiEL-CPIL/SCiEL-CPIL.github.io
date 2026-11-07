def sign_edges(graph, budget):
    import numpy as np
    import time
    from scipy.linalg import eigh

    start = time.time()
    DEADLINE = 84.0

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m

    edge_u = np.array([u for u, v in edges], dtype=int)
    edge_v = np.array([v for u, v in edges], dtype=int)

    # --- Find a spanning tree (BFS) to identify fundamental cycles ---
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

    tree_edge_set = set(parent_edge[1:])
    non_tree = [i for i in range(n_edges) if i not in tree_edge_set]
    n_non_tree = len(non_tree)  # = m + 1 for a prism

    # Precompute base tree adjacency (tree edges always +1)
    A_tree = np.zeros((n_vertices, n_vertices))
    for v in range(1, n_vertices):
        pe = parent_edge[v]
        u = parent[v]
        A_tree[u, v] = 1.0
        A_tree[v, u] = 1.0

    # Precompute for each non-tree edge: (u, v)
    nt_u = edge_u[non_tree]
    nt_v = edge_v[non_tree]

    def spectral_radius(A):
        vals = eigh(A, eigvals_only=True, check_finite=False)
        return float(np.max(np.abs(vals)))

    def is_connected_lift(signs_non_tree):
        # Lift is connected iff at least one non-tree edge has sign -1
        return bool(np.any(signs_non_tree == -1))

    # --- Exact enumeration over all 2^(m+1) switching classes ---
    # Tree edges fixed to +1; non-tree edges vary.
    best_radius = float('inf')
    best_nt = None

    total = 1 << n_non_tree
    # Skip the all-+1 case (disconnected lift)
    for mask in range(1, total):
        if (time.time() - start) > DEADLINE - 20.0:
            break

        signs_nt = np.ones(n_non_tree, dtype=np.int8)
        # set -1 for bits that are 1
        mm = mask
        k = 0
        while mm:
            if mm & 1:
                signs_nt[k] = -1
            mm >>= 1
            k += 1
        # remaining k..n_non_tree-1 stay +1 (already +1)

        if not is_connected_lift(signs_nt):
            continue

        A = A_tree.copy()
        for k in range(n_non_tree):
            s = signs_nt[k]
            u = nt_u[k]
            v = nt_v[k]
            A[u, v] = s
            A[v, u] = s

        r = spectral_radius(A)
        if r < best_radius:
            best_radius = r
            best_nt = signs_nt.copy()

    if best_nt is None:
        # fallback: all +1 except one -1
        best_nt = np.ones(n_non_tree, dtype=np.int8)
        best_nt[0] = -1
        best_radius = float('inf')

    # Build full sign vector from best switching class
    signs = np.ones(n_edges, dtype=np.int8)
    for v in range(1, n_vertices):
        pe = parent_edge[v]
        signs[pe] = 1
    for k, ei in enumerate(non_tree):
        signs[ei] = best_nt[k]

    # --- Polish: local search on the full sign vector ---
    A = np.zeros((n_vertices, n_vertices))
    for i in range(n_edges):
        s = signs[i]
        u = edge_u[i]
        v = edge_v[i]
        A[u, v] = s
        A[v, u] = s

    def lift_connected(A):
        for k in range(n_non_tree):
            if A[nt_u[k], nt_v[k]] == -1:
                return True
        return False

    def local_search(signs, A, cur_r, max_time):
        improved = True
        while improved and (time.time() - start) < max_time:
            improved = False
            for i in range(n_edges):
                if (time.time() - start) > max_time:
                    return signs, cur_r
                old = signs[i]
                u = edge_u[i]
                v = edge_v[i]
                signs[i] = -old
                A[u, v] = -A[u, v]
                A[v, u] = -A[v, u]
                if not lift_connected(A):
                    signs[i] = old
                    A[u, v] = -A[u, v]
                    A[v, u] = -A[v, u]
                    continue
                nr = spectral_radius(A)
                if nr < cur_r - 1e-12:
                    cur_r = nr
                    improved = True
                else:
                    signs[i] = old
                    A[u, v] = -A[u, v]
                    A[v, u] = -A[v, u]
        return signs, cur_r

    if (time.time() - start) < DEADLINE - 6.0:
        signs, best_radius = local_search(signs, A, best_radius, DEADLINE - 4.0)

    # --- Random restarts if time remains ---
    rng = np.random.default_rng(7)
    while (time.time() - start) < DEADLINE - 3.0:
        trial = signs.copy()
        n_flips = int(rng.integers(1, 4))
        for _ in range(n_flips):
            idx = int(rng.integers(0, n_edges))
            trial[idx] *= -1
        tA = np.zeros((n_vertices, n_vertices))
        for i in range(n_edges):
            s = trial[i]
            u = edge_u[i]
            v = edge_v[i]
            tA[u, v] = s
            tA[v, u] = s
        if not lift_connected(tA):
            continue
        tr = spectral_radius(tA)
        if tr < best_radius:
            trial, tr = local_search(trial, tA, tr, DEADLINE - 2.0)
            if tr < best_radius:
                best_radius = tr
                signs = trial
                A = tA

    return signs.astype(int)

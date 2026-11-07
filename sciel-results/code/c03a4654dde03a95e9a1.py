def sign_edges(graph, budget):
    import numpy as np
    import time
    from collections import deque

    deadline = time.time() + 55.0  # safety margin before 60s wall-clock kill

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)

    # Edge index lookup
    idx = {}
    for i, (u, v) in enumerate(edges):
        idx[(min(u, v), max(u, v))] = i

    def key(u, v):
        return (min(u, v), max(u, v))

    # Classify edges
    bot_edges = []
    top_edges = []
    match_edges = []
    for i, (u, v) in enumerate(edges):
        a, b = min(u, v), max(u, v)
        if 0 <= a < m and 0 <= b < m:
            bot_edges.append(i)
        elif a >= m and b >= m:
            top_edges.append(i)
        else:
            match_edges.append(i)

    # Build adjacency for balance check
    adj_with_edge = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj_with_edge[u].append((v, i))
        adj_with_edge[v].append((u, i))

    def is_balanced(sgn):
        """Check if the signed graph is balanced (lift disconnected)."""
        pot = np.zeros(n, dtype=np.int8)
        pot[0] = 1
        dq = deque([0])
        while dq:
            u = dq.popleft()
            pu = pot[u]
            for v, ei in adj_with_edge[u]:
                s = sgn[ei]
                want = int(pu * s)
                if pot[v] == 0:
                    pot[v] = want
                    dq.append(v)
                elif pot[v] != want:
                    return False
        return True

    # Precompute base adjacency for spectral radius
    A_base = np.zeros((n, n))
    for (u, v) in edges:
        A_base[u, v] = 1
        A_base[v, u] = 1

    def spectral_radius(sgn):
        A = A_base.copy()
        for (u, v) in edges:
            s = sgn[idx[key(u, v)]]
            A[u, v] = s
            A[v, u] = s
        w = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(w)))

    def spectral_radius_with_eigvec(sgn):
        """Return (radius, eigenvector) for the dominant eigenvalue."""
        A = A_base.copy()
        for (u, v) in edges:
            s = sgn[idx[key(u, v)]]
            A[u, v] = s
            A[v, u] = s
        w, V = np.linalg.eigh(A)
        # Find the eigenvalue with max absolute value
        max_idx = np.argmax(np.abs(w))
        return float(abs(w[max_idx])), V[:, max_idx]

    def gradient_spectral_radius(sgn):
        """Compute gradient of spectral radius w.r.t. edge signs (continuous relaxation)."""
        A = A_base.copy()
        for (u, v) in edges:
            s = sgn[idx[key(u, v)]]
            A[u, v] = s
            A[v, u] = s
        w, V = np.linalg.eigh(A)
        max_idx = np.argmax(np.abs(w))
        lam = w[max_idx]
        v = V[:, max_idx]
        
        # Gradient of lambda_max w.r.t. A_ij is v_i * v_j
        # For spectral radius = |lambda_max|, if lambda_max > 0, grad = v_i*v_j
        # if lambda_max < 0, grad = -v_i*v_j
        sign_lam = 1.0 if lam >= 0 else -1.0
        grad_A = sign_lam * np.outer(v, v)
        
        # Map back to edge gradient
        grad_edge = np.zeros(E)
        for i, (u, v) in enumerate(edges):
            grad_edge[i] = grad_A[u, v] + grad_A[v, u]
        
        return grad_edge

    best_signs = None
    best_radius = float('inf')

    def consider(sgn):
        nonlocal best_signs, best_radius
        if is_balanced(sgn):
            return
        r = spectral_radius(sgn)
        if r < best_radius:
            best_radius = r
            best_signs = sgn.copy()

    # ---- Phase 1: Gauge-fixed enumeration ----
    if 2 ** (2 * m) <= 500000:
        for code in range(2 ** (2 * m)):
            if time.time() > deadline - 5:
                break
            sgn = np.ones(E, dtype=np.int8)
            for j in range(m):
                bit = (code >> j) & 1
                sgn[top_edges[j]] = 1 if bit == 0 else -1
            for j in range(m):
                bit = (code >> (m + j)) & 1
                sgn[match_edges[j]] = 1 if bit == 0 else -1
            consider(sgn)
    else:
        rng = np.random.default_rng(3)
        for _ in range(10000):
            if time.time() > deadline - 5:
                break
            sgn = np.ones(E, dtype=np.int8)
            for j in range(m):
                sgn[top_edges[j]] = rng.choice([-1, 1])
            for j in range(m):
                sgn[match_edges[j]] = rng.choice([-1, 1])
            consider(sgn)

    if best_signs is None:
        best_signs = np.ones(E, dtype=np.int8)
        best_radius = float('inf')

    # ---- Phase 2: Gradient-based continuous optimization ----
    if time.time() < deadline - 10:
        # Start from current best, relax to continuous
        x = best_signs.astype(np.float64)
        lr = 0.1
        for step in range(500):
            if time.time() > deadline - 10:
                break
            grad = gradient_spectral_radius(x)
            # Gradient descent step
            x = x - lr * grad
            # Clamp to [-1, 1]
            x = np.clip(x, -1.0, 1.0)
            # Decay learning rate
            lr *= 0.995
        
        # Round to nearest sign and check validity
        rounded = np.where(x >= 0, 1, -1).astype(np.int8)
        if not is_balanced(rounded):
            r = spectral_radius(rounded)
            if r < best_radius:
                best_radius = r
                best_signs = rounded.copy()
        
        # Also try rounding with small perturbations
        rng3 = np.random.default_rng(99)
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            perturbed = x + rng3.normal(0, 0.1, size=E)
            perturbed = np.clip(perturbed, -1.0, 1.0)
            rounded2 = np.where(perturbed >= 0, 1, -1).astype(np.int8)
            if not is_balanced(rounded2):
                r = spectral_radius(rounded2)
                if r < best_radius:
                    best_radius = r
                    best_signs = rounded2.copy()

    # ---- Phase 3: Local search refinement ----
    cur = best_signs.copy()
    cur_r = best_radius

    flip_order = np.arange(E, dtype=np.int64)
    np.random.default_rng(42).shuffle(flip_order)

    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for ei in flip_order:
            if time.time() > deadline - 3:
                break
            cur[ei] = -cur[ei]
            if is_balanced(cur):
                cur[ei] = -cur[ei]
                continue
            r = spectral_radius(cur)
            if r < cur_r - 1e-9:
                cur_r = r
                improved = True
                if r < best_radius:
                    best_radius = r
                    best_signs = cur.copy()
            else:
                cur[ei] = -cur[ei]

    # ---- Phase 4: Multi-flip search ----
    if time.time() < deadline - 3:
        rng2 = np.random.default_rng(7)
        cur2 = best_signs.copy()
        cur2_r = spectral_radius(cur2) if not is_balanced(cur2) else float('inf')

        improved2 = True
        while improved2 and time.time() < deadline - 3:
            improved2 = False
            for _ in range(100):
                if time.time() > deadline - 3:
                    break
                i, j = rng2.integers(0, E, size=2)
                if i == j:
                    continue
                cur2[i] = -cur2[i]
                cur2[j] = -cur2[j]
                if is_balanced(cur2):
                    cur2[i] = -cur2[i]
                    cur2[j] = -cur2[j]
                    continue
                r = spectral_radius(cur2)
                if r < cur2_r - 1e-9:
                    cur2_r = r
                    improved2 = True
                    if r < best_radius:
                        best_radius = r
                        best_signs = cur2.copy()
                else:
                    cur2[i] = -cur2[i]
                    cur2[j] = -cur2[j]

    # ---- Phase 5: High-precision polish ----
    if time.time() < deadline - 1:
        # Very tight local search with smaller tolerance
        cur5 = best_signs.copy()
        cur5_r = best_radius
        for _ in range(50):
            if time.time() > deadline - 1:
                break
            best_flip = None
            best_new_r = cur5_r
            for ei in range(E):
                if time.time() > deadline - 1:
                    break
                cur5[ei] = -cur5[ei]
                if is_balanced(cur5):
                    cur5[ei] = -cur5[ei]
                    continue
                r = spectral_radius(cur5)
                if r < best_new_r - 1e-12:
                    best_new_r = r
                    best_flip = ei
                cur5[ei] = -cur5[ei]
            
            if best_flip is not None:
                cur5[best_flip] = -cur5[best_flip]
                cur5_r = best_new_r
                if cur5_r < best_radius:
                    best_radius = cur5_r
                    best_signs = cur5.copy()
            else:
                break

    return np.asarray(best_signs, dtype=np.int8)

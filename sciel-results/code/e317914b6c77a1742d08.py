import time
import random
import math
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def construct():
    n = 40
    m = 401
    start_time = time.time()
    deadline = start_time + 118.0

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_edges_set = set(all_edges)

    def compute_max_book(masks, edges):
        max_b = 0
        for u, v in edges:
            common = masks[u] & masks[v]
            b = common.bit_count()
            if b > max_b:
                max_b = b
        return max_b

    def compute_book_sizes(masks, edges):
        book_sizes = {}
        for u, v in edges:
            common = masks[u] & masks[v]
            book_sizes[(u, v)] = common.bit_count()
        return book_sizes

    def spectral_init():
        """Use Fiedler vector to create a balanced bipartition, then add edges to reach m."""
        # Build a random sparse graph to get a Laplacian
        A = np.zeros((n, n))
        for _ in range(200):
            u, v = random.randint(0, n - 1), random.randint(0, n - 1)
            if u != v:
                A[u, v] = 1
                A[v, u] = 1
        deg = A.sum(axis=1)
        L = np.diag(deg) - A
        # Shift to ensure positive definite
        L = L + 0.1 * np.eye(n)
        try:
            vals, vecs = eigsh(L, k=2, which='SM')
            fiedler = vecs[:, 1]
        except Exception:
            fiedler = np.random.randn(n)
            fiedler -= fiedler.mean()

        # Sort by Fiedler value to create bipartition
        order = np.argsort(fiedler)
        half = n // 2
        left = set(order[:half].tolist())
        right = set(order[half:].tolist())

        # Start with all edges within left and within right (bipartite complement)
        edges = set()
        for u in left:
            for v in left:
                if u < v:
                    edges.add((u, v))
        for u in right:
            for v in right:
                if u < v:
                    edges.add((u, v))

        # Add cross edges to reach m
        cross_edges = [(u, v) for u in left for v in right if u < v]
        random.shuffle(cross_edges)
        for e in cross_edges:
            if len(edges) >= m:
                break
            edges.add(e)

        # If still not enough, add random edges
        if len(edges) < m:
            remaining = [e for e in all_edges if e not in edges]
            random.shuffle(remaining)
            for e in remaining:
                if len(edges) >= m:
                    break
                edges.add(e)

        masks = [0] * n
        for u, v in edges:
            masks[u] |= 1 << v
            masks[v] |= 1 << u

        return edges, masks

    def qp_relaxation(edges, masks, max_iters=50):
        """
        QP relaxation: minimize sum of (book_size(e) - target)^2 over all edges.
        We use a continuous relaxation where each edge variable is in [0,1].
        """
        if time.time() > deadline - 20.0:
            return edges, masks, compute_max_book(masks, edges)

        # Build adjacency matrix
        A = np.zeros((n, n))
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1

        # For each edge (u,v), book size = number of common neighbors = sum_w A[u,w] * A[v,w]
        # We want to minimize max book size. Use a penalty: sum_e book(e)^2
        # This is a quartic in A, but we can linearize by introducing auxiliary variables.

        # Alternative: use a simpler approach - minimize the sum of squares of degrees
        # which correlates with book size.

        # Let's use a different approach: minimize the trace of A^3 (number of triangles)
        # which is related to book sizes.

        # Actually, let's use a direct approach: for each edge, the book size is
        # the number of triangles containing it. We can approximate by minimizing
        # the sum over all edges of (common_neighbors)^2.

        # Build the objective: for each edge (u,v), common = sum_w A[u,w]*A[v,w]
        # We want to minimize sum_{(u,v) in E} common_{uv}^2

        # This is a quartic polynomial in A. We can use scipy.optimize.minimize
        # with a finite difference gradient.

        def objective(x):
            # x is a vector of length n*(n-1)/2, representing the upper triangle of A
            # Reconstruct A
            idx = 0
            A_x = np.zeros((n, n))
            for u in range(n):
                for v in range(u + 1, n):
                    A_x[u, v] = x[idx]
                    A_x[v, u] = x[idx]
                    idx += 1

            # Compute book sizes for all potential edges
            total = 0.0
            for u in range(n):
                for v in range(u + 1, n):
                    if A_x[u, v] > 0.5:  # edge exists
                        common = 0
                        for w in range(n):
                            if w != u and w != v:
                                common += A_x[u, w] * A_x[v, w]
                        total += common * common
            return total

        # Initial point: current edge indicators
        x0 = np.zeros(n * (n - 1) // 2)
        idx = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) in edges:
                    x0[idx] = 1.0
                idx += 1

        # Constraints: sum x = m
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - m}]
        bounds = [(0, 1)] * len(x0)

        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=constraints, options={'maxiter': max_iters, 'ftol': 1e-6})
            if res.success:
                # Project back to graph
                new_edges = set()
                idx = 0
                for u in range(n):
                    for v in range(u + 1, n):
                        if res.x[idx] > 0.5:
                            new_edges.add((u, v))
                        idx += 1

                # Ensure exactly m edges
                if len(new_edges) > m:
                    # Remove edges with highest book size
                    new_masks = [0] * n
                    for u, v in new_edges:
                        new_masks[u] |= 1 << v
                        new_masks[v] |= 1 << u
                    book_sz = compute_book_sizes(new_masks, new_edges)
                    sorted_edges = sorted(new_edges, key=lambda e: -book_sz[e])
                    for e in sorted_edges:
                        if len(new_edges) <= m:
                            break
                        new_edges.remove(e)
                elif len(new_edges) < m:
                    # Add edges with lowest book size
                    new_masks = [0] * n
                    for u, v in new_edges:
                        new_masks[u] |= 1 << v
                        new_masks[v] |= 1 << u
                    remaining = [e for e in all_edges if e not in new_edges]
                    random.shuffle(remaining)
                    for e in remaining:
                        if len(new_edges) >= m:
                            break
                        new_edges.add(e)

                new_masks = [0] * n
                for u, v in new_edges:
                    new_masks[u] |= 1 << v
                    new_masks[v] |= 1 << u

                return new_edges, new_masks, compute_max_book(new_masks, new_edges)
        except Exception:
            pass

        return edges, masks, compute_max_book(masks, edges)

    def fast_local_search(edges, masks, max_iters=50000):
        """Fast local search with edge swaps to minimize max book."""
        if time.time() > deadline - 10.0:
            return edges, masks, compute_max_book(masks, edges)

        edges = set(edges)
        masks = list(masks)
        book_sizes = compute_book_sizes(masks, edges)
        current_max = max(book_sizes.values()) if book_sizes else 0
        best_edges = set(edges)
        best_masks = list(masks)
        best_max = current_max

        non_edges = all_edges_set - edges

        # Precompute for speed
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break

            # Pick a random edge to remove
            edge_list = list(edges)
            a, b = random.choice(edge_list)

            # Pick a random non-edge to add
            if not non_edges:
                break
            non_edge_list = list(non_edges)
            c, d = random.choice(non_edge_list)

            # Compute common neighbors
            common_ab = masks[a] & masks[b]
            common_cd = masks[c] & masks[d]
            b_ab = common_ab.bit_count()
            b_cd = common_cd.bit_count()

            # Quick check: if adding (c,d) would create a book > best_max, skip
            if b_cd > best_max:
                continue

            # Check if swap is valid (no edge exceeds best_max)
            valid = True
            T = common_cd
            while T:
                lsb = T & -T
                w = lsb.bit_length() - 1
                T ^= lsb
                e_cw = (min(c, w), max(c, w))
                e_dw = (min(d, w), max(d, w))
                if e_cw in edges and book_sizes.get(e_cw, 0) + 1 > best_max:
                    valid = False
                    break
                if e_dw in edges and book_sizes.get(e_dw, 0) + 1 > best_max:
                    valid = False
                    break

            if not valid:
                continue

            # Perform swap
            edges.discard((a, b))
            edges.add((c, d))
            masks[a] &= ~(1 << b)
            masks[b] &= ~(1 << a)
            masks[c] |= 1 << d
            masks[d] |= 1 << c

            non_edges.discard((c, d))
            non_edges.add((a, b))

            # Update book sizes
            book_sizes.pop((a, b), None)
            T = common_ab
            while T:
                lsb = T & -T
                w = lsb.bit_length() - 1
                T ^= lsb
                e_aw = (min(a, w), max(a, w))
                e_bw = (min(b, w), max(b, w))
                if e_aw in book_sizes:
                    book_sizes[e_aw] -= 1
                if e_bw in book_sizes:
                    book_sizes[e_bw] -= 1

            T = common_cd
            while T:
                lsb = T & -T
                w = lsb.bit_length() - 1
                T ^= lsb
                e_cw = (min(c, w), max(c, w))
                e_dw = (min(d, w), max(d, w))
                if e_cw in book_sizes:
                    book_sizes[e_cw] += 1
                else:
                    book_sizes[e_cw] = 1
                if e_dw in book_sizes:
                    book_sizes[e_dw] += 1
                else:
                    book_sizes[e_dw] = 1
            book_sizes[(c, d)] = b_cd

            # Recompute max
            current_max = max(book_sizes.values()) if book_sizes else 0

            if current_max < best_max:
                best_max = current_max
                best_edges = set(edges)
                best_masks = list(masks)

            # If we've hit the theoretical lower bound, stop
            if best_max <= 6:
                break

        return best_edges, best_masks, best_max

    def greedy_with_constraint(B, max_attempts=10):
        """Greedy construction with book size constraint B."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 15.0:
                break

            edges = set()
            masks = [0] * n
            book_sizes = {}

            all_edges_shuffled = all_edges[:]
            random.shuffle(all_edges_shuffled)

            for u, v in all_edges_shuffled:
                if len(edges) >= m:
                    break

                common = masks[u] & masks[v]
                b_new = common.bit_count()
                if b_new > B:
                    continue

                # Check that adding (u,v) won't push any existing edge's book over B
                valid = True
                T = common
                while T:
                    lsb = T & -T
                    w = lsb.bit_length() - 1
                    T ^= lsb
                    e_uw = (min(u, w), max(u, w))
                    e_vw = (min(v, w), max(v, w))
                    if book_sizes.get(e_uw, 0) >= B:
                        valid = False
                        break
                    if book_sizes.get(e_vw, 0) >= B:
                        valid = False
                        break

                if not valid:
                    continue

                edges.add((u, v))
                masks[u] |= 1 << v
                masks[v] |= 1 << u

                T = common
                while T:
                    lsb = T & -T
                    w = lsb.bit_length() - 1
                    T ^= lsb
                    e_uw = (min(u, w), max(u, w))
                    e_vw = (min(v, w), max(v, w))
                    book_sizes[e_uw] = book_sizes.get(e_uw, 0) + 1
                    book_sizes[e_vw] = book_sizes.get(e_vw, 0) + 1

                book_sizes[(u, v)] = b_new

            if len(edges) == m:
                return edges, masks

        return None, None

    best_edges = None
    best_masks = None
    best_B = float('inf')

    # Phase 1: Spectral initialization
    if time.time() < deadline - 60.0:
        edges, masks = spectral_init()
        max_b = compute_max_book(masks, edges)
        if max_b < best_B:
            best_B = max_b
            best_edges = edges
            best_masks = masks

    # Phase 2: QP relaxation
    if time.time() < deadline - 40.0 and best_edges is not None:
        qp_edges, qp_masks, qp_max = qp_relaxation(best_edges, best_masks, max_iters=30)
        if qp_max < best_B:
            best_B = qp_max
            best_edges = qp_edges
            best_masks = qp_masks

    # Phase 3: Greedy with constraint
    for B in [7, 8, 9]:
        if time.time() > deadline - 30.0:
            break
        g_edges, g_masks = greedy_with_constraint(B, max_attempts=5)
        if g_edges is not None:
            g_max = compute_max_book(g_masks, g_edges)
            if g_max < best_B:
                best_B = g_max
                best_edges = g_edges
                best_masks = g_masks
            if g_max <= B:
                break

    # Phase 4: Fast local search
    if time.time() < deadline - 15.0:
        if best_edges is None:
            edges = set()
            all_edges_shuffled = all_edges[:]
            random.shuffle(all_edges_shuffled)
            for u, v in all_edges_shuffled:
                if len(edges) >= m:
                    break
                edges.add(u, v) if False else edges.add((u, v))
            masks = [0] * n
            for u, v in edges:
                masks[u] |= 1 << v
                masks[v] |= 1 << u
            best_edges = edges
            best_masks = masks
            best_B = compute_max_book(masks, edges)

        ls_edges, ls_masks, ls_max = fast_local_search(best_edges, best_masks, max_iters=50000)
        if ls_max < best_B:
            best_B = ls_max
            best_edges = ls_edges
            best_masks = ls_masks

    # Phase 5: Another round of QP + local search if time permits
    if time.time() < deadline - 10.0 and best_edges is not None:
        qp_edges, qp_masks, qp_max = qp_relaxation(best_edges, best_masks, max_iters=20)
        if qp_max < best_B:
            best_B = qp_max
            best_edges = qp_edges
            best_masks = qp_masks

        if time.time() < deadline - 5.0:
            ls_edges, ls_masks, ls_max = fast_local_search(best_edges, best_masks, max_iters=20000)
            if ls_max < best_B:
                best_B = ls_max
                best_edges = ls_edges
                best_masks = ls_masks

    if best_edges is None:
        edges = set()
        all_edges_shuffled = all_edges[:]
        random.shuffle(all_edges_shuffled)
        for u, v in all_edges_shuffled:
            if len(edges) >= m:
                break
            edges.add((u, v))
        best_edges = edges

    return [[u, v] for u, v in best_edges]

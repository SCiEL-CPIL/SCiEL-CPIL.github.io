"""Heuristic matrix for ACO-TSP: Local search + geometric priors + spectral gap penalty."""
import numpy as np
import time
import scipy.spatial

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # Safety margin (330s limit, 50s margin)


def _mds_coordinates(D):
    """Recover 2D coordinates from a Euclidean distance matrix using Classical MDS."""
    n = D.shape[0]
    if n < 3:
        return None

    if not np.allclose(D, D.T, atol=1e-8):
        return None
    if np.any(D < -1e-9):
        return None

    J = np.eye(n) - np.ones((n, n)) / n
    D2 = D ** 2
    B = -0.5 * (J @ D2 @ J)
    B = (B + B.T) / 2.0

    try:
        w, V = np.linalg.eigh(B)
    except Exception:
        return None

    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]

    if w[0] < 1e-9:
        return None

    coords = V[:, :2] * np.sqrt(np.maximum(w[:2], 0))

    # Verify reconstruction quality
    rec_D = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2))
    corr = np.corrcoef(rec_D.flatten(), D.flatten())[0, 1]
    if np.isnan(corr) or corr < 0.99:
        return None

    return coords


def _tour_length(tour, D):
    n = len(tour)
    return sum(D[tour[i], tour[(i + 1) % n]] for i in range(n))


def _two_opt(tour, D, max_iters=200):
    """Standard 2-opt improvement."""
    n = len(tour)
    if n < 4:
        return list(tour)

    best_tour = list(tour)
    improved = True
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = best_tour[i], best_tour[i + 1]
                c, d = best_tour[j], best_tour[(j + 1) % n]

                old_len = D[a, b] + D[c, d]
                new_len = D[a, c] + D[b, d]

                if new_len < old_len - 1e-12:
                    best_tour[i + 1:j + 1] = best_tour[i + 1:j + 1][::-1]
                    improved = True
                    break
            if improved:
                break
    return best_tour


def _or_opt(tour, D, max_iters=100):
    """Or-opt: move 1, 2, or 3 consecutive nodes to another position."""
    n = len(tour)
    if n < 5:
        return list(tour)

    best_tour = list(tour)
    best_len = _tour_length(best_tour, D)
    improved = True
    iter_count = 0

    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        for i in range(n):
            for size in [1, 2, 3]:
                if i + size >= n:
                    continue

                # Segment to move
                seg = best_tour[i:i + size]
                # Remove segment
                remaining = best_tour[:i] + best_tour[i + size:]

                # Try inserting at all valid positions
                for j in range(len(remaining)):
                    # Skip positions that would recreate the same tour
                    if j == i and (i + size == n or i == 0):
                        continue
                    if j > i and j <= i + size:
                        continue

                    new_tour = remaining[:j] + seg + remaining[j:]
                    new_len = _tour_length(new_tour, D)

                    if new_len < best_len - 1e-12:
                        best_len = new_len
                        best_tour = new_tour
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return best_tour


def _improve_tour(tour, D):
    """Combined 2-opt and Or-opt improvement with multiple passes."""
    current_tour = list(tour)
    best_tour = list(tour)
    best_len = _tour_length(best_tour, D)

    for _ in range(15):
        prev_len = best_len

        # 2-Opt pass
        current_tour = _two_opt(current_tour, D, max_iters=100)
        curr_len = _tour_length(current_tour, D)
        if curr_len < best_len - 1e-12:
            best_len = curr_len
            best_tour = list(current_tour)

        # Or-opt pass
        current_tour = _or_opt(current_tour, D, max_iters=50)
        curr_len = _tour_length(current_tour, D)
        if curr_len < best_len - 1e-12:
            best_len = curr_len
            best_tour = list(current_tour)

        # Another 2-opt pass
        current_tour = _two_opt(current_tour, D, max_iters=100)
        curr_len = _tour_length(current_tour, D)
        if curr_len < best_len - 1e-12:
            best_len = curr_len
            best_tour = list(current_tour)

        if abs(best_len - prev_len) < 1e-9:
            break

    return best_tour


def _spectral_gap_penalty(D, coords):
    """
    Compute Fiedler vector of the graph Laplacian derived from inverse-distance weights.
    Return a penalty matrix where entries are < 1 for edges connecting nodes with large
    differences in their Fiedler coordinates.
    """
    n = D.shape[0]
    if n < 4:
        return None

    # Build weighted adjacency matrix using inverse distance
    W = np.zeros((n, n))
    mask = D > 1e-9
    W[mask] = 1.0 / D[mask]
    np.fill_diagonal(W, 0.0)

    # Compute degree matrix
    degrees = W.sum(axis=1)
    degrees[degrees == 0] = 1e-9

    # Normalized Laplacian: L_sym = I - D^(-1/2) W D^(-1/2)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L = np.eye(n) - (D_inv_sqrt @ W @ D_inv_sqrt)
    L = (L + L.T) / 2.0  # Symmetrize

    try:
        w, V = np.linalg.eigh(L)
    except Exception:
        return None

    # Fiedler vector is the second eigenvector (smallest non-zero eigenvalue)
    # Sort by eigenvalue
    idx = np.argsort(w)
    w = w[idx]
    V = V[:, idx]

    # The first eigenvector corresponds to eigenvalue 0 (constant vector)
    # Fiedler vector is the next one
    if len(w) < 2:
        return None

    fiedler = V[:, 1]

    # Penalty: dampen edges with large differences in Fiedler coordinates
    # Use a smooth exponential penalty
    penalty = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                penalty[i, j] = 1.0
                continue
            diff = abs(fiedler[i] - fiedler[j])
            # Scale: typical Fiedler values are in [-1, 1], so diff in [0, 2]
            # Dampen strongly for large differences
            penalty[i, j] = np.exp(-2.0 * diff)

    return penalty


def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix.
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]

    if n < 3:
        heuristic = 1.0 / (d + 1e-9) ** 4
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        heuristic = 1.0 / (d + 1e-10) ** 3
        np.fill_diagonal(heuristic, 0.0)
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # 1. Base heuristic: inverse distance
    heuristic = 1.0 / (d + 1e-10)

    # 2. Find a high-quality reference tour
    # Start with Nearest Neighbor from multiple starts
    best_tour = None
    best_len = np.inf

    starts = [0, 1, n // 2, n - 1, n // 3, 2 * n // 3]
    for start in starts:
        if time.time() > _DEADLINE - 15.0:
            break

        visited = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        current = start

        while unvisited:
            min_dist = np.inf
            next_node = -1
            for node in unvisited:
                if d[current, node] < min_dist:
                    min_dist = d[current, node]
                    next_node = node
            visited.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        improved_tour = _improve_tour(visited, d)
        tour_len = _tour_length(improved_tour, d)
        if tour_len < best_len:
            best_len = tour_len
            best_tour = improved_tour

    if best_tour is None:
        # Fallback: simple NN from 0
        visited = [0]
        unvisited = set(range(1, n))
        current = 0
        while unvisited:
            min_dist = np.inf
            next_node = -1
            for node in unvisited:
                if d[current, node] < min_dist:
                    min_dist = d[current, node]
                    next_node = node
            visited.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        best_tour = _improve_tour(visited, d)
        best_len = _tour_length(best_tour, d)

    # 3. Boost edges from the best tour
    # The ACO algorithm will prefer these edges.
    tour_boost = 15.0
    for i in range(n):
        u = best_tour[i]
        v = best_tour[(i + 1) % n]
        heuristic[u, v] *= tour_boost
        heuristic[v, u] *= tour_boost

    # 4. Geometric boosts if MDS succeeded
    coords = _mds_coordinates(d)
    if coords is not None and time.time() < _DEADLINE - 10.0:
        try:
            # Delaunay Triangulation boost
            tri = scipy.spatial.Delaunay(coords)
            delaunay_boost = 5.0
            for simplex in tri.simplices:
                i, j, k = simplex
                heuristic[i, j] *= delaunay_boost
                heuristic[j, i] *= delaunay_boost
                heuristic[j, k] *= delaunay_boost
                heuristic[k, j] *= delaunay_boost
                heuristic[i, k] *= delaunay_boost
                heuristic[k, i] *= delaunay_boost

            # Convex Hull boost
            hull = scipy.spatial.ConvexHull(coords)
            hull_boost = 20.0
            for simplex in hull.simplices:
                i, j = simplex
                heuristic[i, j] *= hull_boost
                heuristic[j, i] *= hull_boost

            # Penalize edges that cross the best tour
            tour_edges = [(best_tour[i], best_tour[(i + 1) % n]) for i in range(n)]

            for i in range(n):
                for j in range(i + 1, n):
                    if i == j:
                        continue

                    pa = coords[i]
                    pb = coords[j]

                    # Check crossing with tour edges
                    crosses = False
                    for (c, dv) in tour_edges:
                        if c == i or c == j or dv == i or dv == j:
                            continue
                        pc = coords[c]
                        pd = coords[dv]

                        def cross(o, a, b):
                            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

                        d1 = cross(pc, pd, pa)
                        d2 = cross(pc, pd, pb)
                        d3 = cross(pa, pb, pc)
                        d4 = cross(pa, pb, pd)

                        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
                           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                            crosses = True
                            break

                    if crosses:
                        heuristic[i, j] *= 0.05
                        heuristic[j, i] *= 0.05

        except Exception:
            pass

    # 5. K-nearest neighbors boost
    k = 4
    knn_boost = 3.0
    for i in range(n):
        dists = d[i].copy()
        dists[i] = np.inf
        nn_indices = np.argpartition(dists, k)[:k]
        for j in nn_indices:
            heuristic[i, j] *= knn_boost
            heuristic[j, i] *= knn_boost

    # 6. Spectral gap penalty (Fiedler vector)
    if coords is not None and time.time() < _DEADLINE - 5.0:
        try:
            penalty = _spectral_gap_penalty(d, coords)
            if penalty is not None:
                heuristic *= penalty
        except Exception:
            pass

    # Ensure diagonal is 0
    np.fill_diagonal(heuristic, 0.0)

    # Final check: non-finite or non-positive become 1e-9
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    return heuristic

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Cover the unit sphere with 50 equal caps.
    Strategy:
    1. Fibonacci spiral initialization.
    2. Fast vectorized Lloyd's relaxation.
    3. Delaunay-based geometric refinement: directly minimize the worst
       spherical triangle's circumradius by moving its vertices along
       geodesic bisectors.
    4. Multiple restarts with perturbations to escape local minima.
    5. Explicit deadline guard for safe return.
    """
    start_time = time.time()
    deadline = start_time + 112.0

    def fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def normalize(pts):
        n = np.linalg.norm(pts, axis=1, keepdims=True)
        n = np.where(n < 1e-14, 1.0, n)
        return pts / n

    def compute_covering_radius_sampled(points, num_test=200000):
        test = fibonacci_sphere(num_test)
        dots = np.clip(test @ points.T, -1.0, 1.0)
        min_dot = np.min(dots, axis=1)
        return np.degrees(np.arccos(np.max(min_dot)))

    def compute_covering_radius_precise(points):
        """Exact covering radius via convex hull facet circumradii."""
        try:
            hull = ConvexHull(points)
        except Exception:
            return compute_covering_radius_sampled(points, 300000)

        max_rad = 0.0
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            e1, e2 = v2 - v1, v3 - v1
            A = np.array([[2*np.dot(e1,e1), 2*np.dot(e1,e2)],
                          [2*np.dot(e1,e2), 2*np.dot(e2,e2)]])
            b = np.array([np.dot(e1,e1), np.dot(e2,e2)])
            try:
                params = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            cc = v1 + params[0]*e1 + params[1]*e2
            ncc = np.linalg.norm(cc)
            if ncc < 1e-12:
                continue
            cc = cc / ncc
            d = np.clip(np.dot(cc, v1), -1.0, 1.0)
            ang = np.arccos(d)
            if ang > max_rad:
                max_rad = ang
        return np.degrees(max_rad)

    def lloyd_iteration(points, test_points, step_size=1.0):
        dots = np.clip(test_points @ points.T, -1.0, 1.0)
        nearest = np.argmax(dots, axis=1)
        n_pts = len(points)
        sums = np.zeros((n_pts, 3))
        np.add.at(sums, nearest, test_points)
        counts = np.bincount(nearest, minlength=n_pts)
        counts_safe = np.where(counts == 0, 1, counts)
        centroids = sums / counts_safe[:, np.newaxis]
        centroids = normalize(centroids)
        empty = counts == 0
        if np.any(empty):
            centroids[empty] = points[empty]
        result = (1 - step_size) * points + step_size * centroids
        return normalize(result)

    def get_all_facet_data(points):
        """Return list of (angular_radius, simplex_indices, circumcenter) for all facets."""
        try:
            hull = ConvexHull(points)
        except Exception:
            return []
        data = []
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            e1, e2 = v2 - v1, v3 - v1
            A = np.array([[2*np.dot(e1,e1), 2*np.dot(e1,e2)],
                          [2*np.dot(e1,e2), 2*np.dot(e2,e2)]])
            b = np.array([np.dot(e1,e1), np.dot(e2,e2)])
            try:
                params = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            cc = v1 + params[0]*e1 + params[1]*e2
            ncc = np.linalg.norm(cc)
            if ncc < 1e-12:
                continue
            cc = cc / ncc
            d = np.clip(np.dot(cc, v1), -1.0, 1.0)
            ang = np.arccos(d)
            data.append((ang, tuple(int(x) for x in simplex), cc))
        return data

    def refine_worst_facets_geodesic(points, num_rounds=60, top_k=8, lr=0.015, beta=50.0):
        """
        Geodesic-bisector refinement with smooth log-sum-exp weighting.
        Uses a soft-max over facet angles to create a smooth surrogate for
        the max, avoiding discontinuous top-k selection.
        """
        pts = points.copy()
        current_lr = lr
        for rnd in range(num_rounds):
            if time.time() > deadline:
                break
            facet_data = get_all_facet_data(pts)
            if not facet_data:
                break

            # Compute smooth weights via log-sum-exp for stability
            angles = np.array([f[0] for f in facet_data])
            # Soft-max with temperature: higher beta = more concentrated on worst
            max_ang = np.max(angles)
            exp_vals = np.exp(beta * (angles - max_ang))
            total = np.sum(exp_vals)
            if total < 1e-30:
                weights = np.ones(len(facet_data)) / len(facet_data)
            else:
                weights = exp_vals / total

            # Also keep a hard top-k floor to ensure worst facets get attention
            sorted_idx = np.argsort(angles)[::-1]
            top_k_set = set(int(x) for x in sorted_idx[:top_k])
            for i in top_k_set:
                weights[i] = max(weights[i], 0.05)
            # Renormalize
            wsum = np.sum(weights)
            if wsum > 0:
                weights /= wsum

            grad = np.zeros_like(pts)
            for i, (ang, simplex, cc) in enumerate(facet_data):
                w = weights[i]
                if w < 1e-8:
                    continue
                for idx in simplex:
                    cur = pts[idx]
                    to_cc = cc - cur
                    radial = np.dot(to_cc, cur) * cur
                    tangent = to_cc - radial
                    nt = np.linalg.norm(tangent)
                    if nt > 1e-12:
                        grad[idx] += w * (tangent / nt)

            gn = np.linalg.norm(grad)
            if gn > 1e-12:
                grad = grad / gn
                pts = pts + current_lr * grad
                pts = normalize(pts)

            # Adaptive lr: reduce if progress stalls, increase slightly if converging well
            current_lr *= 0.985
            if current_lr < 0.002:
                current_lr = 0.002

        return pts

    def high_precision_polish(points, num_rounds=80, beta=100.0, lr_start=0.003):
        """
        Final high-precision polish pass with tighter tolerances and
        very small steps to fine-tune the covering radius.
        """
        pts = points.copy()
        current_lr = lr_start
        prev_score = float('inf')
        for rnd in range(num_rounds):
            if time.time() > deadline - 2.0:
                break
            facet_data = get_all_facet_data(pts)
            if not facet_data:
                break

            angles = np.array([f[0] for f in facet_data])
            max_ang = np.max(angles)
            exp_vals = np.exp(beta * (angles - max_ang))
            total = np.sum(exp_vals)
            if total < 1e-30:
                weights = np.ones(len(facet_data)) / len(facet_data)
            else:
                weights = exp_vals / total

            # Hard top-3 always gets minimum weight
            sorted_idx = np.argsort(angles)[::-1]
            for i in sorted_idx[:3]:
                weights[int(i)] = max(weights[int(i)], 0.1)
            wsum = np.sum(weights)
            if wsum > 0:
                weights /= wsum

            grad = np.zeros_like(pts)
            for i, (ang, simplex, cc) in enumerate(facet_data):
                w = weights[i]
                if w < 1e-8:
                    continue
                for idx in simplex:
                    cur = pts[idx]
                    to_cc = cc - cur
                    radial = np.dot(to_cc, cur) * cur
                    tangent = to_cc - radial
                    nt = np.linalg.norm(tangent)
                    if nt > 1e-12:
                        grad[idx] += w * (tangent / nt)

            gn = np.linalg.norm(grad)
            if gn > 1e-14:
                grad = grad / gn
                pts = pts + current_lr * grad
                pts = normalize(pts)

            # Very gradual decay
            current_lr *= 0.97
            if current_lr < 0.0003:
                current_lr = 0.0003

        return pts

    def separate_close_points(points, min_cos=0.9995):
        """Push apart points that are nearly coincident."""
        pts = points.copy()
        for i in range(N):
            for j in range(i+1, N):
                d = np.dot(pts[i], pts[j])
                if d > min_cos:
                    diff = pts[i] - pts[j]
                    nd = np.linalg.norm(diff)
                    if nd > 1e-10:
                        diff = diff / nd
                        push = 0.005 * (1.0 - d) / (1.0 - min_cos + 1e-9)
                        pts[i] = pts[i] + push * diff
                        pts[j] = pts[j] - push * diff
                        pts[i] = pts[i] / np.linalg.norm(pts[i])
                        pts[j] = pts[j] / np.linalg.norm(pts[j])
        return pts

    # Pre-compute Lloyd test grid
    lloyd_test = fibonacci_sphere(120000)
    rng = np.random.RandomState(12345)

    best_points = None
    best_score = float('inf')

    num_restarts = 0
    max_restarts = 20

    while time.time() < deadline and num_restarts < max_restarts:
        num_restarts += 1

        # Initialize
        if num_restarts == 1:
            points = fibonacci_sphere(N)
        elif num_restarts == 2:
            # Slightly perturbed Fibonacci
            points = fibonacci_sphere(N) + rng.standard_normal((N, 3)) * 0.03
            points = normalize(points)
        else:
            if best_points is not None and rng.random() < 0.6:
                points = best_points + rng.standard_normal((N, 3)) * 0.04
            else:
                points = rng.standard_normal((N, 3))
            points = normalize(points)

        # Phase 1: Lloyd's relaxation
        step_size = 0.7
        for it in range(70):
            if time.time() > deadline:
                break
            points = lloyd_iteration(points, lloyd_test, step_size)
            step_size *= 0.95
            if step_size < 0.015:
                step_size = 0.015

        # Phase 2: Separate close points
        points = separate_close_points(points)

        # Phase 3: Geodesic facet refinement (main refinement) with smooth weights
        points = refine_worst_facets_geodesic(points, num_rounds=55, top_k=8, lr=0.015, beta=40.0)

        # Evaluate
        score = compute_covering_radius_precise(points)

        if score < best_score:
            best_score = score
            best_points = points.copy()

        # Phase 4: If good, do a second, finer refinement pass
        if best_score < 18.5 and time.time() < deadline - 25:
            points = refine_worst_facets_geodesic(points, num_rounds=45, top_k=5, lr=0.008, beta=60.0)
            points = separate_close_points(points, min_cos=0.9999)
            score2 = compute_covering_radius_precise(points)
            if score2 < best_score:
                best_score = score2
                best_points = points.copy()

            # Phase 5: High-precision polish on the best result
            if best_score < 18.45 and time.time() < deadline - 10:
                polished = high_precision_polish(best_points, num_rounds=100, beta=120.0, lr_start=0.002)
                polished = separate_close_points(polished, min_cos=0.99995)
                score3 = compute_covering_radius_precise(polished)
                if score3 < best_score:
                    best_score = score3
                    best_points = polished.copy()

    # Final normalization
    if best_points is not None:
        best_points = normalize(best_points)

    return best_points
# EVOLVE-BLOCK-END

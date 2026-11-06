"""Combined program: explicit construction + geodesic gradient refinement + high-precision polish."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Cover the unit sphere with 50 equal caps.
    Strategy:
    1. High-quality initial guess via Fibonacci spiral + random perturbation.
    2. Fast vectorized Lloyd's relaxation to get a reasonable tiling.
    3. Geodesic gradient descent directly on the convex-hull covering radius
       using a log-sum-exp smooth surrogate for facet weights.
    4. High-precision polish pass with very small steps.
    5. Multiple restarts with different seeds and perturbation levels.
    6. Explicit deadline guard for safe return.
    """
    start_time = time.time()
    deadline = start_time + 140.0

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

    def compute_covering_radius_precise(points):
        """Exact covering radius via convex hull facet circumradii (in degrees)."""
        try:
            hull = ConvexHull(points)
        except Exception:
            test = fibonacci_sphere(200000)
            dots = np.clip(test @ points.T, -1.0, 1.0)
            min_dot = np.min(dots, axis=1)
            return np.degrees(np.arccos(np.max(min_dot)))

        max_rad = 0.0
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            e1, e2 = v2 - v1, v3 - v1
            A = np.array([[2 * np.dot(e1, e1), 2 * np.dot(e1, e2)],
                          [2 * np.dot(e1, e2), 2 * np.dot(e2, e2)]])
            b = np.array([np.dot(e1, e1), np.dot(e2, e2)])
            try:
                params = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            cc = v1 + params[0] * e1 + params[1] * e2
            ncc = np.linalg.norm(cc)
            if ncc < 1e-12:
                continue
            cc = cc / ncc
            d = np.clip(np.dot(cc, v1), -1.0, 1.0)
            ang = np.arccos(d)
            if ang > max_rad:
                max_rad = ang
        return np.degrees(max_rad)

    def get_facet_data(points):
        """
        Return arrays: (facet_radii [F], facet_ccs [F,3], facet_verts [F,3] int).
        """
        try:
            hull = ConvexHull(points)
        except Exception:
            return None, None, None

        radii = []
        ccs = []
        verts = []
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            e1, e2 = v2 - v1, v3 - v1
            A = np.array([[2 * np.dot(e1, e1), 2 * np.dot(e1, e2)],
                          [2 * np.dot(e1, e2), 2 * np.dot(e2, e2)]])
            b = np.array([np.dot(e1, e1), np.dot(e2, e2)])
            try:
                params = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            cc = v1 + params[0] * e1 + params[1] * e2
            ncc = np.linalg.norm(cc)
            if ncc < 1e-12:
                continue
            cc = cc / ncc
            d = np.clip(np.dot(cc, v1), -1.0, 1.0)
            ang = np.arccos(d)
            radii.append(ang)
            ccs.append(cc)
            verts.append(simplex)

        if not radii:
            return None, None, None

        return np.array(radii), np.array(ccs), np.array(verts, dtype=int)

    def refine_geodesic(points, num_rounds=100, beta=50.0, base_lr=0.012):
        """
        Geodesic gradient refinement using a smooth log-sum-exp surrogate.
        For each facet, compute how much moving each of its vertices toward
        the circumcenter reduces the circumradius. Use a soft max over facets
        to get a differentiable objective, then take a gradient step.
        """
        pts = points.copy()
        current_lr = base_lr
        for rnd in range(num_rounds):
            if time.time() > deadline - 2.0:
                break

            radii, ccs, verts = get_facet_data(pts)
            if radii is None:
                break

            # Compute smooth weights via log-sum-exp for stability
            max_r = np.max(radii)
            exp_vals = np.exp(beta * (radii - max_r))
            total_w = np.sum(exp_vals)
            if total_w < 1e-30:
                weights = np.ones(len(radii)) / len(radii)
            else:
                weights = exp_vals / total_w

            # Also keep a hard top-k floor to ensure worst facets get attention
            sorted_idx = np.argsort(radii)[::-1]
            top_k_set = set(int(x) for x in sorted_idx[:8])
            for i in top_k_set:
                weights[i] = max(weights[i], 0.05)
            # Renormalize
            wsum = np.sum(weights)
            if wsum > 0:
                weights /= wsum

            # Compute gradient: for each vertex, sum contributions from incident facets
            grad = np.zeros_like(pts)
            for f in range(len(radii)):
                w = weights[f]
                if w < 1e-10:
                    continue
                cc = ccs[f]
                for idx in verts[f]:
                    cur = pts[idx]
                    # Direction from cur toward cc on the tangent plane
                    to_cc = cc - cur
                    radial_comp = np.dot(to_cc, cur) * cur
                    tangent = to_cc - radial_comp
                    nt = np.linalg.norm(tangent)
                    if nt > 1e-12:
                        grad[idx] += w * (tangent / nt)

            gn = np.linalg.norm(grad)
            if gn > 1e-14:
                pts = pts + current_lr * (grad / gn)
                pts = normalize(pts)

            # Adaptive lr: reduce if progress stalls
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
        for rnd in range(num_rounds):
            if time.time() > deadline - 2.0:
                break

            radii, ccs, verts = get_facet_data(pts)
            if radii is None:
                break

            # Compute smooth weights via log-sum-exp for stability
            max_r = np.max(radii)
            exp_vals = np.exp(beta * (radii - max_r))
            total_w = np.sum(exp_vals)
            if total_w < 1e-30:
                weights = np.ones(len(radii)) / len(radii)
            else:
                weights = exp_vals / total_w

            # Hard top-3 always gets minimum weight
            sorted_idx = np.argsort(radii)[::-1]
            for i in sorted_idx[:3]:
                weights[int(i)] = max(weights[int(i)], 0.1)
            wsum = np.sum(weights)
            if wsum > 0:
                weights /= wsum

            # Compute gradient
            grad = np.zeros_like(pts)
            for f in range(len(radii)):
                w = weights[f]
                if w < 1e-10:
                    continue
                cc = ccs[f]
                for idx in verts[f]:
                    cur = pts[idx]
                    to_cc = cc - cur
                    radial_comp = np.dot(to_cc, cur) * cur
                    tangent = to_cc - radial_comp
                    nt = np.linalg.norm(tangent)
                    if nt > 1e-12:
                        grad[idx] += w * (tangent / nt)

            gn = np.linalg.norm(grad)
            if gn > 1e-14:
                pts = pts + current_lr * (grad / gn)
                pts = normalize(pts)

            # Adaptive lr: reduce if progress stalls
            current_lr *= 0.98
            if current_lr < 0.0005:
                current_lr = 0.0005

        return pts

    def separate_close_points(points, min_cos=0.9995):
        """Push apart points that are nearly coincident."""
        pts = points.copy()
        for _ in range(3):
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.dot(pts[i], pts[j])
                    if d > min_cos:
                        diff = pts[i] - pts[j]
                        nd = np.linalg.norm(diff)
                        if nd > 1e-10:
                            diff = diff / nd
                            push = 0.002 * (1.0 - d) / (1.0 - min_cos + 1e-9)
                            pts[i] = pts[i] + push * diff
                            pts[j] = pts[j] - push * diff
                            pts[i] = pts[i] / np.linalg.norm(pts[i])
                            pts[j] = pts[j] / np.linalg.norm(pts[j])
        return pts

    # Pre-compute Lloyd test grid
    lloyd_test = fibonacci_sphere(100000)

    best_points = None
    best_score = float('inf')

    rng = np.random.RandomState(42)

    max_restarts = 20
    restart = 0

    while time.time() < deadline and restart < max_restarts:
        restart += 1

        # Initialization
        if restart == 1:
            points = fibonacci_sphere(N)
        elif restart == 2:
            points = fibonacci_sphere(N) + rng.standard_normal((N, 3)) * 0.02
            points = normalize(points)
        elif restart <= 5:
            points = fibonacci_sphere(N) + rng.standard_normal((N, 3)) * (0.03 + 0.02 * (restart - 3))
            points = normalize(points)
        elif best_points is not None and rng.random() < 0.7:
            points = best_points + rng.standard_normal((N, 3)) * 0.03
            points = normalize(points)
        else:
            points = rng.standard_normal((N, 3))
            points = normalize(points)

        # Phase 1: Lloyd's relaxation
        step_size = 0.5
        for it in range(60):
            if time.time() > deadline - 20:
                break
            points = lloyd_iteration(points, lloyd_test, step_size)
            step_size *= 0.95
            if step_size < 0.02:
                step_size = 0.02

        # Phase 2: Separate close points
        points = separate_close_points(points)

        # Phase 3: Geodesic refinement
        points = refine_geodesic(points, num_rounds=80, beta=50.0, base_lr=0.012)
        points = separate_close_points(points, min_cos=0.9998)

        # Evaluate
        score = compute_covering_radius_precise(points)

        if score < best_score:
            best_score = score
            best_points = points.copy()

        # Phase 4: Fine-tune if we're getting close to known best
        if best_score < 18.4 and time.time() < deadline - 30:
            points = high_precision_polish(points, num_rounds=60, beta=100.0, lr_start=0.003)
            points = separate_close_points(points, min_cos=0.9999)
            score2 = compute_covering_radius_precise(points)
            if score2 < best_score:
                best_score = score2
                best_points = points.copy()

    # Final normalization
    if best_points is not None:
        best_points = normalize(best_points)
    else:
        best_points = fibonacci_sphere(N)

    return best_points
# EVOLVE-BLOCK-END

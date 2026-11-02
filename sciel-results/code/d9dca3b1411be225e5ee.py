import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety

    # Precompute all triples
    all_triples = list(combinations(range(N), 3))
    tri_idx = np.array(all_triples, dtype=np.int32)  # shape (286, 3)
    n_triples = len(all_triples)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 0.0, 0.0
        if area_hull < 1e-9:
            return 0.0, 0.0
        # Vectorized area computation
        i, j, k = tri_idx[:, 0], tri_idx[:, 1], tri_idx[:, 2]
        v1x = points[j, 0] - points[i, 0]
        v1y = points[j, 1] - points[i, 1]
        v2x = points[k, 0] - points[i, 0]
        v2y = points[k, 1] - points[i, 1]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        score = min_area / area_hull
        return score, min_area, areas

    def _force_directed_step(points, lr=0.01, iters=50):
        """
        Structural change: instead of soft-min annealing, directly push
        points away from the worst-case (smallest area) triangles using
        a repulsive force along the triangle edges.
        """
        pts = points.copy()
        for it in range(iters):
            if time.time() > start_time + time_limit - 10:
                break
            i, j, k = tri_idx[:, 0], tri_idx[:, 1], tri_idx[:, 2]
            v1x = pts[j, 0] - pts[i, 0]
            v1y = pts[j, 1] - pts[i, 1]
            v2x = pts[k, 0] - pts[i, 0]
            v2y = pts[k, 1] - pts[i, 1]
            cross = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * cross

            # Find the worst (smallest area) triangle and push its vertices apart
            min_idx = np.argmin(areas)
            ia, ja, ka = all_triples[min_idx]

            # Compute direction to increase area of this triangle
            # Increase area by moving each vertex outward from the centroid
            centroid = (pts[ia] + pts[ja] + pts[ka]) / 3.0

            # Force: move each vertex away from centroid, scaled by current area deficit
            # Use the area of the triangle as a proxy for how "tight" it is
            # Push proportionally to 1/area to give stronger force to smaller triangles
            force_scale = lr / max(areas[min_idx], 1e-10)
            force_scale = min(force_scale, 0.05)  # Cap the force

            for p_idx in [ia, ja, ka]:
                direction = pts[p_idx] - centroid
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    pts[p_idx] += force_scale * direction / norm

        return pts

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    best_points = None
    best_score = -float('inf')

    # Generate diverse starting configurations
    configs = []

    # Regular 13-gon
    configs.append(_regular_polygon(13, radius=1.0))

    # Ring configurations
    center_pt = np.array([[0.0, 0.0]])
    outer_12 = _regular_polygon(12, radius=1.0)
    configs.append(np.vstack([center_pt, outer_12]))

    inner_6 = _regular_polygon(6, radius=0.5)
    outer_6 = _regular_polygon(6, radius=1.0)
    configs.append(np.vstack([center_pt, inner_6, outer_6]))

    two_centers = np.array([[-0.15, 0.0], [0.15, 0.0]])
    outer_11 = _regular_polygon(11, radius=1.0)
    configs.append(np.vstack([two_centers, outer_11]))

    # Concentric rings: 1 + 4 + 8
    c1 = np.array([[0.0, 0.0]])
    r4 = _regular_polygon(4, radius=0.4)
    r8 = _regular_polygon(8, radius=0.85)
    configs.append(np.vstack([c1, r4, r8]))

    # Concentric rings: 1 + 3 + 9
    c2 = np.array([[0.0, 0.0]])
    r3 = _regular_polygon(3, radius=0.35)
    r9 = _regular_polygon(9, radius=0.85)
    configs.append(np.vstack([c2, r3, r9]))

    # Concentric rings: 1 + 5 + 7
    c3 = np.array([[0.0, 0.0]])
    r5 = _regular_polygon(5, radius=0.45)
    r7 = _regular_polygon(7, radius=0.9)
    configs.append(np.vstack([c3, r5, r7]))

    # Perturbed regular polygons
    for seed in range(5):
        rng = np.random.RandomState(seed)
        pert = _regular_polygon(13, radius=1.0) + rng.randn(13, 2) * 0.03
        configs.append(pert)

    for config in configs:
        if time.time() > start_time + time_limit - 40:
            break

        # Evaluate initial config
        try:
            s, min_area, areas = compute_score(config)
            if s > best_score:
                best_score = s
                best_points = config.copy()
        except Exception:
            continue

        # Force-directed optimization
        current = config.copy()
        for lr in [0.03, 0.02, 0.01, 0.005]:
            if time.time() > start_time + time_limit - 30:
                break
            for _ in range(30):
                if time.time() > start_time + time_limit - 25:
                    break
                current = _force_directed_step(current, lr=lr, iters=20)
                try:
                    s, _, _ = compute_score(current)
                    if s > best_score:
                        best_score = s
                        best_points = current.copy()
                except Exception:
                    break

        # L-BFGS-B polish with a different objective: maximize min area directly
        # using a log-sum-exp approximation with higher beta
        if time.time() > start_time + time_limit - 20:
            break

        def _neg_soft_min(x, beta=200.0):
            pts = x.reshape(-1, 2)
            try:
                hull = ConvexHull(pts)
                area_hull = hull.volume
            except Exception:
                return 1e6
            if area_hull < 1e-9:
                return 1e6
            i, j, k = tri_idx[:, 0], tri_idx[:, 1], tri_idx[:, 2]
            v1x = pts[j, 0] - pts[i, 0]
            v1y = pts[j, 1] - pts[i, 1]
            v2x = pts[k, 0] - pts[i, 0]
            v2y = pts[k, 1] - pts[i, 1]
            cross = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * cross
            min_a = np.min(areas)
            shifted = -beta * (areas - min_a)
            max_s = np.max(shifted)
            lse = max_s + np.log(np.sum(np.exp(shifted - max_s)))
            soft_min = min_a - lse / beta
            return -(soft_min / area_hull)

        try:
            result = opt.minimize(
                _neg_soft_min,
                current.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 100}
            )
            opt_pts = result.x.reshape(-1, 2)
            s, _, _ = compute_score(opt_pts)
            if s > best_score:
                best_score = s
                best_points = opt_pts.copy()
        except Exception:
            pass

    # Nelder-Mead final polish
    if best_points is not None and time.time() < start_time + time_limit - 10:
        def _hard_neg(x):
            pts = x.reshape(-1, 2)
            try:
                s, _, _ = compute_score(pts)
                return -s
            except Exception:
                return 1e6

        try:
            res = opt.minimize(
                _hard_neg,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'adaptive': True, 'xatol': 1e-13, 'fatol': 1e-15}
            )
            nm_pts = res.x.reshape(-1, 2)
            s, _, _ = compute_score(nm_pts)
            if s > best_score:
                best_score = s
                best_points = nm_pts.copy()
        except Exception:
            pass

    # Additional force-directed rounds from the best point
    if best_points is not None and time.time() < start_time + time_limit - 5:
        for lr in [0.01, 0.005, 0.002]:
            if time.time() > start_time + time_limit - 3:
                break
            current = best_points.copy()
            for _ in range(20):
                if time.time() > start_time + time_limit - 3:
                    break
                current = _force_directed_step(current, lr=lr, iters=10)
                try:
                    s, _, _ = compute_score(current)
                    if s > best_score:
                        best_score = s
                        best_points = current.copy()
                except Exception:
                    break

    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)

    return best_points

# EVOLVE-BLOCK-END

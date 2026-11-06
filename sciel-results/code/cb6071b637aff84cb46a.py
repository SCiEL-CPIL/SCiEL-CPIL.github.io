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
    1. Multiple diverse initializations (Fibonacci, random, perturbed).
    2. Fast Lloyd's relaxation with adaptive step size.
    3. Direct facet-based optimization: minimize the maximum circumradius
       of the spherical Delaunay triangles using gradient descent on the
       worst facets with careful tangent-space projection.
    4. Local search with simulated-annealing-like acceptance to escape
       local minima.
    5. Explicit deadline guard for safe return.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

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

    def compute_covering_radius_precise(points):
        """Exact covering radius via convex hull facet circumradii."""
        try:
            hull = ConvexHull(points)
        except Exception:
            # Fallback: dense sampling
            test = fibonacci_sphere(300000)
            dots = np.clip(test @ points.T, -1.0, 1.0)
            min_dot = np.min(dots, axis=1)
            return np.degrees(np.arccos(np.max(min_dot)))

        max_rad = 0.0
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            e1, e2 = v2 - v1, v3 - v1
            A = np.array([[2*np.dot(e1, e1), 2*np.dot(e1, e2)],
                          [2*np.dot(e1, e2), 2*np.dot(e2, e2)]])
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
        """Return list of (angular_radius, simplex_indices, circumcenter) for all facets."""
        try:
            hull = ConvexHull(points)
        except Exception:
            return []
        data = []
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            e1, e2 = v2 - v1, v3 - v1
            A = np.array([[2*np.dot(e1, e1), 2*np.dot(e1, e2)],
                          [2*np.dot(e1, e2), 2*np.dot(e2, e2)]])
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
            data.append((ang, tuple(int(x) for x in simplex), cc))
        return data

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

    def optimize_facets(points, num_rounds=200, lr_start=0.02, lr_end=0.001,
                        top_k=10, temperature=1.0):
        """
        Optimize points to minimize the maximum facet circumradius.
        Uses weighted gradient descent focusing on the worst facets.
        """
        pts = points.copy()
        current_lr = lr_start
        lr_decay = (lr_end / lr_start) ** (1.0 / max(num_rounds, 1))

        for rnd in range(num_rounds):
            if time.time() > deadline - 5.0:
                break

            facet_data = get_facet_data(pts)
            if not facet_data:
                break

            angles = np.array([f[0] for f in facet_data])
            if len(angles) == 0:
                break

            # Weight by how close each facet is to the maximum
            max_ang = np.max(angles)
            # Soft weighting: facets closer to max get higher weight
            weights = np.exp(-temperature * (max_ang - angles))
            weights /= np.sum(weights)

            # Ensure top-k facets get minimum weight
            sorted_idx = np.argsort(angles)[::-1]
            for i in sorted_idx[:top_k]:
                weights[int(i)] = max(weights[int(i)], 0.05)
            weights /= np.sum(weights)

            # Compute gradient: move each vertex toward its facet's circumcenter
            grad = np.zeros_like(pts)
            for i, (ang, simplex, cc) in enumerate(facet_data):
                w = weights[i]
                if w < 1e-10:
                    continue
                for idx in simplex:
                    cur = pts[idx]
                    to_cc = cc - cur
                    # Project to tangent space
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

            current_lr *= lr_decay
            if current_lr < lr_end:
                current_lr = lr_end

        return pts

    def local_search(points, num_trials=50, perturbation_scale=0.02):
        """
        Local search: perturb points slightly and keep if it improves the score.
        """
        rng = np.random.RandomState(int(time.time() * 1000) % 100000)
        pts = points.copy()
        current_score = compute_covering_radius_precise(pts)

        for trial in range(num_trials):
            if time.time() > deadline - 10.0:
                break

            # Random perturbation
            trial_pts = pts + rng.standard_normal((N, 3)) * perturbation_scale
            trial_pts = normalize(trial_pts)

            trial_score = compute_covering_radius_precise(trial_pts)

            # Accept if better or with some probability (simulated annealing)
            if trial_score < current_score:
                pts = trial_pts
                current_score = trial_score
            elif rng.random() < 0.1:  # 10% chance to accept worse
                pts = trial_pts
                current_score = trial_score

            # Gradually reduce perturbation
            perturbation_scale *= 0.95

        return pts, current_score

    # Pre-compute Lloyd test grid
    lloyd_test = fibonacci_sphere(150000)
    rng = np.random.RandomState(42)

    best_points = None
    best_score = float('inf')

    num_restarts = 0
    max_restarts = 15

    while time.time() < deadline - 15.0 and num_restarts < max_restarts:
        num_restarts += 1

        # Diverse initializations
        if num_restarts == 1:
            points = fibonacci_sphere(N)
        elif num_restarts == 2:
            points = fibonacci_sphere(N) + rng.standard_normal((N, 3)) * 0.05
            points = normalize(points)
        elif num_restarts == 3:
            points = rng.standard_normal((N, 3))
            points = normalize(points)
        else:
            if best_points is not None and rng.random() < 0.7:
                points = best_points + rng.standard_normal((N, 3)) * 0.03
            else:
                points = rng.standard_normal((N, 3))
            points = normalize(points)

        # Phase 1: Lloyd's relaxation
        step_size = 0.8
        for it in range(80):
            if time.time() > deadline - 20.0:
                break
            points = lloyd_iteration(points, lloyd_test, step_size)
            step_size *= 0.96
            if step_size < 0.02:
                step_size = 0.02

        # Phase 2: Facet-based optimization
        points = optimize_facets(points, num_rounds=150, lr_start=0.015,
                                 lr_end=0.0005, top_k=10, temperature=2.0)

        # Evaluate
        score = compute_covering_radius_precise(points)

        if score < best_score:
            best_score = score
            best_points = points.copy()

        # Phase 3: Local search on the best result if time permits
        if best_score < 18.6 and time.time() < deadline - 30.0:
            refined, refined_score = local_search(best_points, num_trials=40,
                                                  perturbation_scale=0.015)
            if refined_score < best_score:
                best_score = refined_score
                best_points = refined

        # Phase 4: Second optimization pass on the best result
        if best_score < 18.5 and time.time() < deadline - 20.0:
            points2 = optimize_facets(best_points, num_rounds=100, lr_start=0.008,
                                      lr_end=0.0002, top_k=8, temperature=3.0)
            score2 = compute_covering_radius_precise(points2)
            if score2 < best_score:
                best_score = score2
                best_points = points2

    # Final: one more polish if time allows
    if best_points is not None and time.time() < deadline - 5.0:
        final = optimize_facets(best_points, num_rounds=80, lr_start=0.005,
                                lr_end=0.0001, top_k=5, temperature=5.0)
        final_score = compute_covering_radius_precise(final)
        if final_score < best_score:
            best_points = final
            best_score = final_score

    # Final normalization
    if best_points is not None:
        best_points = normalize(best_points)

    return best_points
# EVOLVE-BLOCK-END

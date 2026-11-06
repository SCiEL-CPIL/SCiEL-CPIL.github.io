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
    1. Start with a Fibonacci spiral.
    2. Fast vectorized Lloyd's relaxation on a large test grid.
    3. Precise convex-hull covering radius evaluation.
    4. Worst-facet targeting refinement to directly shrink the largest empty caps.
    5. Multiple restarts with random perturbations to escape local minima.
    6. Explicit deadline guard to return best result before timeout.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius_sampled(points, num_test_points=100000):
        """
        Estimate covering radius by sampling many points on the sphere.
        """
        test_points = fibonacci_sphere(num_test_points)
        dot_products = test_points @ points.T
        dot_products = np.clip(dot_products, -1.0, 1.0)
        min_dot = np.min(dot_products, axis=1)
        angles = np.arccos(min_dot)
        max_angle = np.max(angles)
        return np.degrees(max_angle)

    def compute_covering_radius_precise(points):
        """
        Compute the covering radius using the convex hull approach.
        For each triangular facet, compute the circumcenter and check
        if it lies on the sphere (i.e., the facet's circumcircle passes
        through the origin's antipode or the facet is 'empty').
        """
        try:
            hull = ConvexHull(points)
        except Exception:
            return compute_covering_radius_sampled(points, 200000)

        max_circumradius = 0.0
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]

            e1 = v2 - v1
            e2 = v3 - v1

            A_mat = np.array([
                [2.0 * np.dot(e1, e1), 2.0 * np.dot(e1, e2)],
                [2.0 * np.dot(e1, e2), 2.0 * np.dot(e2, e2)]
            ])
            b_vec = np.array([np.dot(e1, e1), np.dot(e2, e2)])

            try:
                params = np.linalg.solve(A_mat, b_vec)
            except np.linalg.LinAlgError:
                continue

            circumcenter = v1 + params[0] * e1 + params[1] * e2
            norm_cc = np.linalg.norm(circumcenter)
            if norm_cc < 1e-10:
                continue
            circumcenter_norm = circumcenter / norm_cc

            # Check that the circumcenter is on the "far side" of the facet
            # (i.e., the cap centered at circumcenter_norm is empty)
            dot_val = np.clip(np.dot(circumcenter_norm, v1), -1.0, 1.0)
            angular_radius = np.arccos(dot_val)

            if angular_radius > max_circumradius:
                max_circumradius = angular_radius

        return np.degrees(max_circumradius)

    def lloyd_iteration(points, test_points, step_size=1.0):
        """One iteration of Lloyd's algorithm (vectorized)."""
        dot_products = test_points @ points.T
        dot_products = np.clip(dot_products, -1.0, 1.0)
        nearest = np.argmax(dot_products, axis=1)

        n_points = len(points)
        sums = np.zeros((n_points, 3))
        np.add.at(sums, nearest, test_points)

        counts = np.bincount(nearest, minlength=n_points)

        # Avoid division by zero
        counts_safe = np.where(counts == 0, 1, counts)

        centroids = sums / counts_safe[:, np.newaxis]

        # Normalize centroids
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        new_points = centroids / norms

        # Handle empty clusters (keep old point)
        empty_mask = counts == 0
        if np.any(empty_mask):
            new_points[empty_mask] = points[empty_mask]

        # Interpolate for stability
        result = (1 - step_size) * points + step_size * new_points
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return result / norms

    def get_worst_facets(points, top_k=10):
        """
        Return the top_k worst facets (largest circumradius) with their
        circumcenters and vertex indices.
        """
        try:
            hull = ConvexHull(points)
        except Exception:
            return []

        facet_data = []
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]

            e1 = v2 - v1
            e2 = v3 - v1

            A_mat = np.array([
                [2.0 * np.dot(e1, e1), 2.0 * np.dot(e1, e2)],
                [2.0 * np.dot(e1, e2), 2.0 * np.dot(e2, e2)]
            ])
            b_vec = np.array([np.dot(e1, e1), np.dot(e2, e2)])

            try:
                params = np.linalg.solve(A_mat, b_vec)
            except np.linalg.LinAlgError:
                continue

            cc = v1 + params[0] * e1 + params[1] * e2
            norm_cc = np.linalg.norm(cc)
            if norm_cc < 1e-10:
                continue
            cc_norm = cc / norm_cc

            dot_val = np.clip(np.dot(cc_norm, v1), -1.0, 1.0)
            angular_radius = np.arccos(dot_val)

            facet_data.append((angular_radius, tuple(simplex), cc_norm))

        if not facet_data:
            return []

        facet_data.sort(key=lambda x: x[0], reverse=True)
        return facet_data[:top_k]

    def refine_worst_facets(points, num_steps=30, lr=0.02):
        """
        Directly target the worst facets by moving vertices toward their
        circumcenters (projected onto the tangent plane).
        """
        for step in range(num_steps):
            if time.time() > deadline:
                break

            worst = get_worst_facets(points, top_k=5)
            if not worst:
                break

            gradient = np.zeros_like(points)
            for radius, simplex, cc_norm in worst:
                weight = radius / np.pi
                for idx in simplex:
                    current = points[idx]
                    to_cc = cc_norm - current
                    radial_component = np.dot(to_cc, current) * current
                    tangent_dir = to_cc - radial_component
                    norm_td = np.linalg.norm(tangent_dir)
                    if norm_td > 1e-10:
                        gradient[idx] += weight * tangent_dir / norm_td

            grad_norm = np.linalg.norm(gradient)
            if grad_norm > 1e-10:
                gradient = gradient / grad_norm
                points = points + lr * gradient
                norms = np.linalg.norm(points, axis=1, keepdims=True)
                norms = np.where(norms < 1e-10, 1.0, norms)
                points = points / norms

        return points

    def separate_close_points(points, min_angle=0.05):
        """Separate points that are too close together."""
        cos_min = np.cos(min_angle)
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dot_val = np.dot(points[i], points[j])
                if dot_val > cos_min:
                    diff = points[i] - points[j]
                    norm_diff = np.linalg.norm(diff)
                    if norm_diff > 1e-10:
                        diff_norm = diff / norm_diff
                        push = 0.01 * (1 - dot_val) / (1 - cos_min)
                        points[i] = points[i] + push * diff_norm
                        points[j] = points[j] - push * diff_norm
                        # Re-normalize
                        points[i] = points[i] / np.linalg.norm(points[i])
                        points[j] = points[j] / np.linalg.norm(points[j])
        return points

    # Pre-generate test points for Lloyd's algorithm
    num_lloyd_test = 200000
    lloyd_test_points = fibonacci_sphere(num_lloyd_test)

    rng = np.random.RandomState(42)

    # Main loop: multiple restarts
    best_points = None
    best_score = float('inf')

    num_restarts = 0
    max_restarts = 20

    while time.time() < deadline and num_restarts < max_restarts:
        num_restarts += 1

        # Initialize
        if num_restarts == 1:
            points = fibonacci_sphere(N)
        else:
            # Random restart or perturbation of previous best
            if best_points is not None and rng.random() < 0.5:
                points = best_points + rng.standard_normal((N, 3)) * 0.05
            else:
                points = rng.standard_normal((N, 3))
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms

        # Lloyd's relaxation
        step_size = 0.5
        for it in range(100):
            if time.time() > deadline:
                break
            points = lloyd_iteration(points, lloyd_test_points, step_size)
            step_size *= 0.95
            if step_size < 0.01:
                step_size = 0.01

        # Separate close points
        points = separate_close_points(points)

        # Refine worst facets
        points = refine_worst_facets(points, num_steps=50, lr=0.01)

        # Evaluate
        score = compute_covering_radius_precise(points)

        if score < best_score:
            best_score = score
            best_points = points.copy()

        # If we have a good result, do more refinement
        if best_score < 18.5 and time.time() < deadline - 20:
            # Additional Lloyd + facet refinement
            for it in range(50):
                if time.time() > deadline:
                    break
                points = lloyd_iteration(points, lloyd_test_points, 0.1)
            points = refine_worst_facets(points, num_steps=50, lr=0.005)
            score = compute_covering_radius_precise(points)
            if score < best_score:
                best_score = score
                best_points = points.copy()

    # Final normalization
    if best_points is not None:
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / norms

    return best_points
# EVOLVE-BLOCK-END
